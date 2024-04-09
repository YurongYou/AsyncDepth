import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pyquaternion
from pyquaternion import Quaternion

pio.renderers.default = 'notebook'
ptc_layout_config = {
    'title': {
        'text': 'test vis LiDAR',
        'font': {
            'size': 20,
            'color': 'rgb(150,150,150)',
        },
        'xanchor': 'left',
        'yanchor': 'top'},
    'paper_bgcolor': 'rgb(255,255,255)',
    'width': 1500,
    'height': 1000,
    'margin': {
        'l': 20,
        'r': 20,
        'b': 20,
        't': 20
    },
    'legend': {
        'font': {
            'size': 20,
            'color': 'rgb(150,150,150)',
        },
        'itemsizing': 'constant'
    },
    "hoverlabel": {
        "namelength": -1,
    },
    'showlegend': False,
    'scene': {
        'aspectmode': 'manual',
        'aspectratio': {'x': 0.75, 'y': 0.75, 'z': 0.05},
        'camera': {'eye': {'x': 0, 'y': 0, 'z': 0.5}},
        'xaxis': {'color': 'rgb(150,150,150)',
                  'dtick': 10,
                  'gridcolor': 'rgb(100,100,100)',
                  'range': [-150, 150],
                  'showbackground': False,
                  'showgrid': True,
                  'showline': False,
                  'showticklabels': True,
                  'tickmode': 'linear',
                  'tickprefix': 'x:'},
        'yaxis': {'color': 'rgb(150,150,150)',
                  'dtick': 10,
                  'gridcolor': 'rgb(100,100,100)',
                  'range': [-150, 150],
                  'showbackground': False,
                  'showgrid': True,
                  'showline': False,
                  'showticklabels': True,
                  'tickmode': 'linear',
                  'tickprefix': 'y:'},
        'zaxis': {'color': 'rgb(150,150,150)',
                  'dtick': 10,
                  'gridcolor': 'rgb(100,100,100)',
                  'range': [-10, 10],
                  'showbackground': False,
                  'showgrid': True,
                  'showline': False,
                  'showticklabels': True,
                  'tickmode': 'linear',
                  'tickprefix': 'z:'}},
}


def get_lidar(ptc, name='LiDAR', size=0.8):
    return [go.Scatter3d(
        x=ptc[:, 0],
        y=ptc[:, 1],
        z=ptc[:, 2],
        mode='markers',
        marker_size=size,
        name=name)]


def compute_box(trans_mat, shape):
    w, l, h = shape
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners]).T
    return transform_points(corners_3d, trans_mat)


def get_linemarks(trans_mat, shape):
    corners = compute_box(trans_mat, shape)
    mid_front = (corners[0] + corners[1]) / 2
    mid_left = (corners[0] + corners[3]) / 2
    mid_right = (corners[1] + corners[2]) / 2
    corners = np.vstack(
        (corners, np.vstack([mid_front, mid_left, mid_right])))
    idx = [0, 8, 9, 10, 8, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]
    return corners[idx, :]


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def get_bbox_lidar(box, name='bbox', color='yellow', width=3):
    x, y, z, dx, dy, dz, heading = box
    l, w, h = dx, dy, dz
    trans_mat = np.eye(4, dtype=np.float32)
    trans_mat[:3, 3] = np.array((x, y, z))
    trans_mat[:3, :3] = rotz(heading)
    markers = get_linemarks(trans_mat, (w, l, h))
    return go.Scatter3d(
        mode='lines',
        x=markers[:, 0],
        y=markers[:, 1],
        z=markers[:, 2],
        line=dict(color=color, width=width),
        name=name)

def get_linemarks_from_corners(corners):
    corners = corners[[0,4,7,3,1,5,6,2]]
    mid_front = (corners[0] + corners[1]) / 2
    mid_left = (corners[0] + corners[3]) / 2
    mid_right = (corners[1] + corners[2]) / 2
    corners = np.vstack(
        (corners, np.vstack([mid_front, mid_left, mid_right])))
    idx = [0, 8, 9, 10, 8, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]
    return corners[idx, :]


def get_bbox_lidar_from_corners(corners, name='bbox', color='yellow', width=3):
    markers = get_linemarks_from_corners(corners)
    return go.Scatter3d(
        mode='lines',
        x=markers[:, 0],
        y=markers[:, 1],
        z=markers[:, 2],
        line=dict(color=color, width=width),
        name=name)


def showvelo(lidar, labels=None, predictions=None, size=0.8):
    gt_bboxes = [] if labels is None else [get_bbox_lidar_from_corners(
        label, name=f'gt_bbox_{i}', color='lightgreen') for i, label in enumerate(labels)]
    pred_bboxes = [] if predictions is None else [get_bbox_lidar_from_corners(
        pred, name=f'pred_bbox_{i}', color='red') for i, pred in enumerate(predictions)]
    fig = go.Figure(data=get_lidar(lidar, size=size)
                    + gt_bboxes + pred_bboxes, layout=ptc_layout_config)
    return fig


def cart2hom(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1), dtype=np.float32)))
    return pts_3d_hom


def transform_points(pts_3d_ref, Tr):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(Tr)).reshape(-1, 4)[:, 0:3]


def subsample(points, nbr=200000):
    return points[np.random.permutation(points.shape[0])[:nbr]]


def create_rot_matrix(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = pyquaternion.Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = translation
    return matrix


def cart2hom(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1), dtype=np.float32)))
    return pts_3d_hom


def transform_points(pts_3d_ref, Tr):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(Tr)).reshape(-1, 4)[:, 0:3]
