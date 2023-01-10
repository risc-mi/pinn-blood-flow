import torch
from torch.utils.data import DataLoader, Dataset
from modulus.architecture.fourier_net import FourierNetArch
from modulus.architecture.modified_fourier_net import ModifiedFourierNetArch
from modulus.architecture.dgm import DGMArch
from modulus.architecture.multiplicative_filter_net import MultiplicativeFilterNetArch
from modulus.architecture.siren import SirenArch

import numpy as np
from sympy import Symbol, sqrt, Max

import modulus
from modulus.hydra import to_absolute_path, to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.csv_utils.csv_rw import csv_to_dict
from modulus.continuous.solvers.solver import Solver
from modulus.continuous.domain.domain import Domain
from modulus.continuous.constraints.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)
from modulus.continuous.validator.validator import PointwiseValidator
from modulus.continuous.monitor.monitor import PointwiseMonitor
from modulus.continuous.inferencer.inferencer import PointwiseInferencer
from modulus.key import Key
from modulus.PDES.navier_stokes import NavierStokes
from modulus.PDES.basic import NormalDotVec
from modulus.geometry.tessellation.tessellation import Tessellation

import stl
from stl import mesh
from scipy.spatial import ConvexHull
import math
import pandas as pd
import sympy


@modulus.main(config_path="conf", config_name="conf")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    # path definitions
    point_path = to_absolute_path("./stl_files")
    path_inlet = point_path + "/inlet.stl"
    dict_path_outlet = {'path_outlet1': point_path + "/outlet1.stl",
                        'path_outlet2': point_path + "/outlet2.stl"
                        }
    path_noslip = point_path + "/wall.stl"
    path_interior = point_path + "/closed.stl"
    path_outlet_combined = point_path + '/outlet_combined.stl'
    path_integral = point_path + "/inlet.stl"

    # create and save combined outlet stl
    def combined_stl(meshes, save_path="./combined.stl"):
        combined = mesh.Mesh(np.concatenate([m.data for m in meshes]))
        combined.save(save_path, mode=stl.Mode.ASCII)

    meshes = [mesh.Mesh.from_file(file_) for file_ in dict_path_outlet.values()]
    combined_stl(meshes, path_outlet_combined)

    # read stl files to make geometry
    inlet_mesh = Tessellation.from_stl(path_inlet, airtight=True)
    dict_outlet = {}
    for idx_, key_ in enumerate(dict_path_outlet):
        dict_outlet['outlet'+str(idx_)+'_mesh'] = Tessellation.from_stl(dict_path_outlet[key_], airtight=True)
    noslip_mesh = Tessellation.from_stl(path_noslip, airtight=True)
    interior_mesh = Tessellation.from_stl(path_interior, airtight=True)
    outlet_combined_mesh = Tessellation.from_stl(path_outlet_combined, airtight=True)
    integral_mesh = Tessellation.from_stl(path_integral, airtight=True)

    # params
    # blood density
    rho = 1.050
    # dynamic viscosity [Pa s]
    mu = 0.00385
    # kinematic viscosity [m**2/s]; kin. viscosity  = dynamic viscosity / rho
    nu = mu / rho
    # velocity in center of inlet (parabolic profile)
    inlet_vel = 0.3

    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh.translate([-c for c in center])
        mesh.scale(scale)

    # normalize invars
    def normalize_invar(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale ** dims
        return invar

    # geometry scaling
    scale = 1  # turn off scaling

    # center of overall geometry
    center = (5.61, 0, 0)
    print('Overall geometry center: ', center)

    # scale and center the geometry files
    normalize_mesh(inlet_mesh, center, scale)
    for idx_, key_ in enumerate(dict_outlet):
        normalize_mesh(dict_outlet[key_], center, scale)
    normalize_mesh(noslip_mesh, center, scale)
    normalize_mesh(interior_mesh, center, scale)
    integral_pos = (4,0,0)
    normalize_mesh(integral_mesh, (center[0]-integral_pos[0],center[1]-integral_pos[1],center[2]-integral_pos[2]), scale)
    
    # center of inlet in original coordinate system
    inlet_center_abs = (0, 0, 0)
    print("inlet_center_abs:", inlet_center_abs)

    # scale end center the inlet center
    inlet_center = list((np.array(inlet_center_abs) - np.array(center)) * scale)
    print("inlet_center:", inlet_center)

    # inlet normal vector; should point into the cylinder, not outwards
    inlet_normal = (1, 0, 0)
    print("inlet_normal:", inlet_normal)

    # inlet area
    inlet_area = (0.5**2 * np.pi) * (scale**2)

    # inlet radius
    inlet_radius = 0.5
    print("inlet_radius:", inlet_radius)
    
    # Volumetric flow (= to mass flow for incompressible fluid) at inlet
    # (2*pi*vmax) * integrate{r*(1-r^2/R^2) dr from r=0 to R}
    inlet_volumetric_flow = inlet_vel/2. * inlet_area
    print("Volumetric flow at inlet:", inlet_volumetric_flow)

    # inlet velocity profile
    def circular_parabola(x, y, z, center, normal, radius, max_vel):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]

        distance = sqrt(centered_x ** 2 + centered_y ** 2 + centered_z ** 2)
        parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    # make aneurysm domain
    domain = Domain()

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu * scale, rho=rho, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
        layer_size=256,
        nr_layers=10
    )
    nodes = (
            ns.make_nodes()
            + normal_dot_vel.make_nodes()
            + [flow_net.make_node(name="flow_network", jit=cfg.jit)]
    )

    # add constraints to solver
    # inlet
    u, v, w = circular_parabola(
        Symbol("x"),
        Symbol("y"),
        Symbol("z"),
        center=inlet_center,
        normal=inlet_normal,
        radius=inlet_radius,
        max_vel=inlet_vel
    )
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    for idx_, key_ in enumerate(dict_outlet):
        outlet = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=dict_outlet[key_],
            outvar={"p": 0},
            batch_size=cfg.batch_size.outlet,
        )
        domain.add_constraint(outlet, "outlet"+str(idx_))

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
    )
    domain.add_constraint(interior, "interior")

    # Integral Continuity 1
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=dict_outlet['outlet0_mesh'],
        outvar={"normal_dot_vel": 0.11780972450961724 * scale**2}, # (2*pi*vmax) * integrate{r*(1-r^2/R^2) dr from r=0 to R} 
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        num_workers=1,
        lambda_weighting={"normal_dot_vel": 0.1},
    )
    domain.add_constraint(integral_continuity, "integral_continuity_1")

    # Integral Continuity 2
    integral_continuity_2 = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_mesh,
        outvar={"normal_dot_vel": -1* 0.11780972450961724 * scale**2}, #inlet is oriented towards (-1,0,0), but flow is in (+1,0,0)
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        num_workers=1,
        lambda_weighting={"normal_dot_vel": 0.1},
    )
    domain.add_constraint(integral_continuity_2, "integral_continuity_2")

    # add inferencer
    inferencer_derivatives_boundary = PointwiseInferencer(
        noslip_mesh.sample_boundary(250000),
        ["p", "u", "v", "w",
         "u__x", "u__y", "u__z",
         "v__x", "v__y", "v__z",
         "w__x", "w__y", "w__z",
         "p__x", "p__y", "p__z",
         "u__x__x", "u__y__y", "u__z__z",
         "v__x__x", "v__y__y", "v__z__z",
         "w__x__x", "w__y__y", "w__z__z",
         "p__x__x", "p__y__y", "p__z__z",
         "u__x__y", "u__x__z", "u__y__z",
         "v__x__y", "v__x__z", "v__y__z",
         "w__x__y", "w__x__z", "w__y__z",
         "p__x__y", "p__x__z", "p__y__z",
         "normal_x", "normal_y", "normal_z", "normal_dot_vel"
         ],
        nodes=nodes,
    )
    domain.add_inferencer(inferencer_derivatives_boundary, "inferencer_derivatives_boundary")

    # add inferencer
    inferencer_derivatives_interior = PointwiseInferencer(
        noslip_mesh.sample_interior(250000),
        ["p", "u", "v", "w",
         "u__x", "u__y", "u__z",
         "v__x", "v__y", "v__z",
         "w__x", "w__y", "w__z",
         "p__x", "p__y", "p__z",
         "u__x__x", "u__y__y", "u__z__z",
         "v__x__x", "v__y__y", "v__z__z",
         "w__x__x", "w__y__y", "w__z__z",
         "p__x__x", "p__y__y", "p__z__z",
         "u__x__y", "u__x__z", "u__y__z",
         "v__x__y", "v__x__z", "v__y__z",
         "w__x__y", "w__x__z", "w__y__z",
         "p__x__y", "p__x__z", "p__y__z",
         "sdf__x", "sdf__y", "sdf__z", "sdf"
         ],
        nodes=nodes,
    )
    domain.add_inferencer(inferencer_derivatives_interior, "inferencer_derivatives_interior")

    # add validation data (CFD data from MODSIM)
    mapping = {
        "Points:0": "x",
        "Points:1": "y",
        "Points:2": "z",
        "U:0": "u",
        "U:1": "v",
        "U:2": "w",
        "p": "p",
        "wss": "u__y"
    }
    #modsim entire geometry
    modsim_var = csv_to_dict(to_absolute_path("modsim/modsim_wfenz_hiresbl.csv"), mapping)
    modsim_invar = {key: value for key, value in modsim_var.items() if key in ["x", "y", "z"]}
    modsim_invar = normalize_invar(modsim_invar, center, scale, dims=3)
    modsim_outvar = {key: value for key, value in modsim_var.items() if key in ["u", "v", "w", "p", "u__y"]}
    modsim_validator = PointwiseValidator(modsim_invar, modsim_outvar, nodes, batch_size=4096)
    domain.add_validator(modsim_validator, "modsim_hiresbl_validator")

    #modsim plane slice
    modsim_var = csv_to_dict(to_absolute_path("modsim/modsim_plane_slice.csv"), mapping)
    modsim_invar = {key: value for key, value in modsim_var.items() if key in ["x", "y", "z"]}
    modsim_invar = normalize_invar(modsim_invar, center, scale, dims=3)
    modsim_outvar = {key: value for key, value in modsim_var.items() if key in ["u", "v", "w", "p", "u__y"]}
    modsim_validator = PointwiseValidator(modsim_invar, modsim_outvar, nodes, batch_size=4096)
    domain.add_validator(modsim_validator, "modsim_plane_slice_validator")

    # add pressure monitor
    pressure_inlet = PointwiseMonitor(
        inlet_mesh.sample_boundary(1024),
        output_names=["p"],
        metrics={"pressure_inlet": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(pressure_inlet)

    # add pressure monitor
    pressure_outlet0 = PointwiseMonitor(
        dict_outlet['outlet0_mesh'].sample_boundary(1024),
        output_names=["p"],
        metrics={"pressure_outlet0": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(pressure_outlet0)

    # add pressure monitor
    pressure_outlet1 = PointwiseMonitor(
        dict_outlet['outlet1_mesh'].sample_boundary(1024),
        output_names=["p"],
        metrics={"pressure_outlet1": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(pressure_outlet1)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
