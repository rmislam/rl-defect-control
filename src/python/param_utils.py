import os
dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = dir_path + '/..'
import numpy as np
rng = np.random.default_rng(42)

def readParam(paramName, paramFilename=src_path + '/cpp/parameters.c'):
    foundParam = False
    paramValue = None

    with open(paramFilename, 'r') as f:
        for line in f:
            tokens = line.strip().split()

            if len(tokens) > 0 and tokens[0] == '#define':
                if tokens[1] == paramName:
                    foundParam = True
                    paramValue = tokens[2]

    if foundParam:
        return convertToCorrectType(paramValue)
    
    raise KeyError(f'Param with name {paramName} could not be found in parameters file {paramFilename}')

def readAllParams(paramFilename=src_path + '/cpp/parameters.c'):
    params = {}

    with open(paramFilename, 'r') as f:
        for line in f:
            tokens = line.strip().split()

            if len(tokens) > 0 and tokens[0] == '#define':
                paramName, paramVal = tokens[1:3]
                params[paramName] = convertToCorrectType(paramVal)

    return params

def convertToCorrectType(paramString):
    try:
        if paramString.isdigit():
            return int(paramString)
        else:
            paramFloat = float(paramString)
            return paramFloat
    except ValueError:
        return paramString.strip('"')

def generateSampleIndices():
    I = readParam('I')
    J = readParam('J')
    GEOM_NAME = readParam('GEOM_NAME')
    sample_fraction = readParam('VIZ_SAMPLE_FRAC')
    sample_indices = np.sort(rng.permutation(I * J)[:int(sample_fraction * I * J)])
    filtered_sample_indices = []

    # only keep sample indices that don't lie within obstacles
    for sample_index in sample_indices:
        x = sample_index % I
        y = np.floor(sample_index / I).astype(int)

        if GEOM_NAME == 'FREE':
            filtered_sample_indices.append(sample_index)
        elif GEOM_NAME == 'SINGLE_CHANNEL':
            chan_bot_frac = 9 / 21
            chan_top_frac = 12 / 21

            if y >= chan_bot_frac * J and y <= chan_top_frac * J:
                filtered_sample_indices.append(sample_index)
        elif GEOM_NAME == 'T_JUNCTION':
            h_chan_bot_frac = 9 / 21
            h_chan_top_frac = 12 / 21
            v_chan_left_frac = 18 / 21

            if x >= v_chan_left_frac * I or (y >= h_chan_bot_frac * J and y <= h_chan_top_frac * J):
                filtered_sample_indices.append(sample_index)
        elif GEOM_NAME == 'CROSS_JUNCTION':
            h_chan_bot_frac = 9 / 21
            h_chan_top_frac = 12 / 21
            v_chan_left_frac = 9 / 21
            v_chan_right_frac = 12 / 21

            if ((x >= v_chan_left_frac * I and x <= v_chan_right_frac * I) or
                (y >= h_chan_bot_frac * J and y <= h_chan_top_frac * J)):
                filtered_sample_indices.append(sample_index)
        elif GEOM_NAME in ['CROSS_MAZE_CW', 'CROSS_MAZE_CCW']:
            north_h_chan_top_frac = 24 / 33
            north_h_chan_bot_frac = 21 / 33

            west_v_chan_left_frac = 9 / 33
            west_v_chan_right_frac = 12 / 33

            south_h_chan_top_frac = 12 / 33
            south_h_chan_bot_frac = 9 / 33

            east_v_chan_left_frac = 21 / 33
            east_v_chan_right_frac = 24 / 33

            if ((x >= west_v_chan_left_frac * I and x <= west_v_chan_right_frac * I) or
                (x >= east_v_chan_left_frac * I and x <= east_v_chan_right_frac * I) or
                (y >= north_h_chan_bot_frac * J and y <= north_h_chan_top_frac * J) or
                (y >= south_h_chan_bot_frac * J and y <= south_h_chan_top_frac * J)):
                filtered_sample_indices.append(sample_index)
        else:
            raise ValueError(f'Unsupported geometry name {GEOM_NAME}')

    return filtered_sample_indices