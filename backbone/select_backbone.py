from resnet_2d3d import * 

def select_resnet(network, track_running_stats=True, in_channels=3):
    param = {'feature_size': 1024}
    if network == 'resnet18':
        model = resnet18_2d3d_full(track_running_stats=track_running_stats, in_channels=in_channels)
        param['feature_size'] = 256
    elif network == 'resnet34':
        model = resnet34_2d3d_full(track_running_stats=track_running_stats, in_channels=in_channels)
        param['feature_size'] = 256
    elif network == 'resnet50':
        model = resnet50_2d3d_full(track_running_stats=track_running_stats, in_channels=in_channels)
    elif network == 'resnet101':
        model = resnet101_2d3d_full(track_running_stats=track_running_stats, in_channels=in_channels)
    elif network == 'resnet152':
        model = resnet152_2d3d_full(track_running_stats=track_running_stats, in_channels=in_channels)
    elif network == 'resnet200':
        model = resnet200_2d3d_full(track_running_stats=track_running_stats, in_channels=in_channels)
    else: raise IOError('model type is wrong')

    return model, param