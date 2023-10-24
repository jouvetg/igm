#!/usr/bin/env python3
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import numpy as np
import os, sys, shutil
import json
from types import SimpleNamespace
import xarray as xr

# create HTML layout for app
fig = go.Figure()
app = Dash(__name__)
app.layout = html.Div([
    # dropdown menu for property
    html.Div([dcc.Dropdown(['thickness [m]', 'velocity [m/a]', 'SMB [m]'],
                           'thickness [m]', id='property', )],
             style={'width': '100%', 'display': 'inline-block'}),
    # 3D surface plot
    dcc.Graph(id='mnt_surface', figure=fig),
], style={'font-family': 'monospace', 'font-size': 'x-large'})

@app.callback(
    Output('mnt_surface', 'figure'),
    Input('property', 'value'))
def updata_graph(property):

    # load params
    path_to_json_saved = 'params_saved.json'
    with open(path_to_json_saved, 'r') as json_file:
        json_text = json_file.read()
    params = json.loads(json_text, object_hook=lambda d: SimpleNamespace(**d))

    # read output.nc
    ds = xr.open_dataset(os.path.join(params.working_dir, "output.nc"), engine="netcdf4")

    # get attributes from ds
    bedrock = np.array(ds.topg[0])
    time = np.array(ds.time)
    lat_range = np.array(ds.x)
    lon_range = np.array(ds.y)
    glacier_surfaces = np.array(ds.usurf)
    thicknesses = np.array(ds.thk)
    velocities = np.array(ds.velsurf_mag)
    smbs = np.array(ds.smb)

    # choose property that is displayed on the glacier surface
    if property == "thickness [m]":
        property_maps = thicknesses
        color_scale = "Blues"
        max_property_map = np.max(property_maps)
        min_property_map = np.min(property_maps)
    elif property == "velocity [m/a]":
        property_maps = velocities
        color_scale = "magma"
        max_property_map = np.max(property_maps)
        min_property_map = np.min(property_maps)
    else:
        property_maps = smbs
        color_scale = "rdbu"
        max_property_map = np.max(property_maps)
        min_property_map = np.min(property_maps)
        max_dis = np.max([abs(max_property_map), abs(min_property_map)])
        max_property_map = max_dis
        min_property_map = -max_dis

    # make edges equal so that it looks like a volume
    max_bedrock = np.max(bedrock)
    min_bedrock = np.min(bedrock)
    bedrock[0, :] = min_bedrock
    bedrock[-1, :] = min_bedrock
    bedrock[:, 0] = min_bedrock
    bedrock[:, -1] = min_bedrock

    # create time frames for slider
    res = 1
    frames = []
    slider_steps = []
    for i, year in enumerate(time):

        # update elevation data and property map
        property_map = property_maps[i]
        glacier_surface = glacier_surfaces[i]
        glacier_surface[thicknesses[i] < 1] = None

        # create 3D surface plots with property as surface color
        surface_fig = go.Surface(z=glacier_surface[::res, ::res], x=lat_range[::res], y=lon_range[::res],
                                 colorscale=color_scale, cmax=max_property_map, cmin=min_property_map,
                                 surfacecolor=property_map[::res, ::res], showlegend=True, name='glacier surface',
                                 colorbar=dict(title=property, titleside='right'), )

        # create 3D bedrock plots
        bedrock_fig = go.Surface(z=bedrock[::res, ::res], x=lat_range[::res], y=lon_range[::res], colorscale='gray',
                                 opacity=1, showlegend=True, name='bedrock', cmax=max_bedrock, cmin=0,
                                 colorbar=dict(x=-.2, title="elevation [m]", titleside='right'))

        # create frame
        frame = {"data": [bedrock_fig, surface_fig], "name": int(year)}
        frames.append(frame)

        # add slider step
        slider_step = {"args": [[int(year)], {"frame": {"duration": 0, "redraw": True}, }],
                       "label": str(int(year)), "method": "animate"}
        slider_steps.append(slider_step)

    # define slider with layout
    sliders_dict = {"active": 0, "yanchor": "top", "xanchor": "left",
                    "currentvalue": {"font": {"size": 20}, "prefix": "Year:", "visible": True, "xanchor": "right"},
                    "transition": {"duration": 0, "easing": "cubic-in-out"}, "pad": {"b": 10, "t": 50},
                    "len": 0.9, "x": 0.1, "y": 0, "steps": slider_steps
                    }

    # define figure layout
    try:
        title = params.oggm_RGI_ID
    except:
        title = "glacier"

    fig_dict = dict(data=frames[0]['data'],
                    frames=frames,
                    layout=dict(width=1000, height=800,
                                sliders=[sliders_dict],
                                title= title,
                                legend={"orientation": "h", "yanchor": "bottom", "xanchor": "left"},
                                scene=dict(zaxis=dict(showbackground=True,
                                                      showticklabels=False, title=""),
                                           xaxis=dict(showbackground=False,
                                                      showticklabels=True, visible=True, title="Longitude"),
                                           yaxis=dict(showbackground=False,
                                                      showticklabels=True, visible=True, title="Latitude")),
                                scene_aspectratio=dict(x=1, y=1, z=.3),
                                scene_camera_eye=dict(x=1, y=-1, z=2),
                                updatemenus=[dict(buttons=[dict(args=[None, {"frame": {"duration": 200, "redraw": True},
                                                                             "fromcurrent": True,
                                                                             "transition": {"duration": 0,
                                                                                            "easing": "quadratic-in-out"}}],
                                                                label="Play",
                                                                method="animate"),
                                                           dict(
                                                               args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                              "mode": "immediate",
                                                                              "transition": {"duration": 0}}],
                                                               label="Pause",
                                                               method="animate")],
                                                  direction="left", pad={"r": 10, "t": 87}, showactive=False,
                                                  type="buttons", x=0.1, xanchor="right", y=0, yanchor="top")]
                                ))
    # create figure
    fig = go.Figure(fig_dict)
    return fig

def params_anim_plotly(parser):
    pass


def initialize_anim_plotly(params, state):
    pass


def update_anim_plotly(params, state):
    pass


def finalize_anim_plotly(params, state):
    # start dash app
    app.run_server(debug=False)



if __name__ == '__main__':

    finalize_anim_plotly(None, None)
