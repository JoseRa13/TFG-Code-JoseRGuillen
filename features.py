import json
import time
from enum import Enum

import pandas as pd
from rfpimp import *


class State(Enum):
    MENU = 1
    IN_puzzle = 2
    puzzle_COMPLETED = 3


tutorial_puzzles = ['1. One Box', '2. Separated Boxes', '3. Rotate a Pyramid', '4. Match Silhouettes',
                    '5. Removing Objects', '6. Stretch a Ramp', '7. Max 2 Boxes', '8. Combine 2 Ramps',
                    '9. Scaling Round Objects']

player_puzzle_events = ['ws-create_shape', 'ws-delete_shape', 'ws-select_shape', 'ws-deselect_shape',
                        'ws-select_shape_add', 'ws-move_shape', 'ws-rotate_shape', 'ws-scale_shape',
                        'ws-mode_change', 'ws-click_nothing', 'ws-rotate_view', 'ws-check_solution',
                        'ws-toggle_paint_display', 'ws-palette_change', 'ws-paint',
                        'ws-toggle_snapshot_display', 'ws-snapshot', 'ws-undo_action', 'ws-redo_action']

list_action_events = ['ws-move_shape', 'ws-rotate_shape', 'ws-scale_shape',
                      'ws-check_solution', 'ws-undo_action', 'ws-redo_action',
                      'ws-rotate_view', 'ws-snapshot', 'ws-mode_change',
                      'ws-create_shape', 'ws-select_shape', 'ws-delete_shape', 'ws-select_shape_add']

events_not_affected_by_undo = ['ws-snapshot', 'ws-check_solution', 'ws-click_nothing', 'ws-click_disabled',
                               'ws-toggle_paint_display', 'ws-toggle_snapshot_display']

puzzles_solutions_folder = 'C:/Users/joser/OneDrive/TFG/Data/Fall19studylevels-final'

puzzles_solutions_files = {
    'Bull Market': 'annie1.json',
    'Stranger Shapes': 'annie2.json',
    'Zzz': 'annie3.json',
    'More Than Meets Your Eye': 'annie4.json',
    'Orange Dance': 'annie5.json',
    'Bear Market': 'annie6.json',
    'Ramp Up and Can It': 'annie7.json',
    'Tall and Small': 'annie8.json',
    'Sugar Cones': 'annie9.json',
    'Square Cross-Sections': 'public1.json',
    'Bird Fez': 'public2.json',
    'Pi Henge': 'public3.json',
    '45-Degree Rotations': 'public4.json',
    'Pyramids are Strange': 'public5.json',
    'Boxes Obscure Spheres': 'public6.json',
    'Object Limits': 'public7.json',
    'Not Bird': 'public8.json',
    'Angled Silhouette': 'public9.json',
    '1. One Box': 'tutorial1.json',
    '2. Separated Boxes': 'tutorial2.json',
    '3. Rotate a Pyramid': 'tutorial3.json',
    '4. Match Silhouettes': 'tutorial4.json',
    '5. Removing Objects': 'tutorial5.json',
    '6. Stretch a Ramp': 'tutorial6.json',
    '7. Max 2 Boxes': 'tutorial7.json',
    '8. Combine 2 Ramps': 'tutorial8.json',
    '9. Scaling Round Objects': 'tutorial9.json',
    'Warm Up': 'teaser1.json',
    'Unnecessary': 'teaser2.json',
    'Few Clues': 'teaser3.json',
}

data_path = ''
puzzle_data_path_export = ''
features_path_export = ''

puzzle_set_id_to_num = {'None': 0, 'Basic': 1, 'Intermediate': 2, 'Advanced': 3}

user_active_threshold = 60


def load_data(path):
    df_events = pd.read_csv(path,
                            parse_dates=['time'],
                            date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f%z'),
                            dtype={'session_id': object, 'type': object, 'data': object},
                            sep=';')

    df_events['group'] = [json.loads(x)['group'] if 'group' in json.loads(x).keys() else '' for x in df_events['data']]
    df_events['user'] = [json.loads(x)['user'] if 'user' in json.loads(x).keys() else '' for x in df_events['data']]

    # removing those rows where we don't have a group and a user that is not guest
    df_events = df_events[((df_events['group'] != '') & (df_events['user'] != '') & (df_events['user'] != 'guest'))]
    df_events['full_user'] = df_events['group'] + '~' + df_events['user']
    df_events = df_events.sort_values('time')
    return df_events


def extract_puzzle_features(data_events):
    """
    :type data_events: pd.DataFrame
    """
    puzzles_data = dict()
    for full_user in data_events['full_user'].unique():
        user_events = data_events[data_events['full_user'] == full_user]
        user_events = user_events.sort_values(['session_id', 'time'])
        puzzles_data[full_user] = dict()
        active_puzzle = None
        previous_event = None
        for index, event in user_events.iterrows():
            if event['type'] == 'ws-start_level':
                active_puzzle = json.loads(event['data'])['task_id']
                previous_event = event
                if active_puzzle == 'Sandbox':
                    continue
                if active_puzzle not in puzzles_data[full_user].keys():
                    puzzles_data[full_user][active_puzzle] = dict()
                    puzzles_data[full_user][active_puzzle]['is_completed'] = 0
                    puzzles_data[full_user][active_puzzle]['time'] = 0
                    puzzles_data[full_user][active_puzzle]['n_times_submitted'] = 0

            if active_puzzle is None or active_puzzle == 'Sandbox' or puzzles_data[full_user][active_puzzle]['is_completed'] == 1:
                continue

            if previous_event is None:
                previous_event = event

            if event['type'] in ['ws-exit_to_menu', 'ws-disconnect', 'ws-create_user', 'ws-login_user',
                                 'ws-start_game']:
                active_puzzle = None
                continue

            if event['type'] == 'ws-check_solution':
                puzzles_data[full_user][active_puzzle]['n_times_submitted'] += 1

            if event['type'] == 'ws-puzzle_complete':
                puzzles_data[full_user][active_puzzle]['is_completed'] = 1

                seconds_elapsed = (event['time'] - previous_event['time']).total_seconds()
                puzzles_data[full_user][active_puzzle]['time'] += seconds_elapsed
                active_puzzle = None
                continue

            seconds_elapsed = (event['time'] - previous_event['time']).total_seconds()
            # we consider that the user is active if the last action was done in the last user_active_threshold seconds
            if seconds_elapsed < user_active_threshold:
                puzzles_data[full_user][active_puzzle]['time'] += seconds_elapsed

            previous_event = event

    user_data = []
    for full_user in puzzles_data.keys():
        for puzzle in puzzles_data[full_user].keys():
            user_data.append(
                {
                    'full_user': full_user,
                    'user': full_user.split('~')[1],
                    'group': full_user.split('~')[0],
                    'puzzle': puzzle,
                    'is_completed': puzzles_data[full_user][puzzle]['is_completed'],
                    'n_times_submitted': puzzles_data[full_user][puzzle]['n_times_submitted'],
                    'time': puzzles_data[full_user][puzzle]['time']
                }
            )
    puzzles_df = pd.DataFrame.from_dict(user_data)
    return puzzles_df


def extract_feautres(data_events):
    features = {
        'puzzle': [],
        'puzzle_difficulty': [],
        'n_puzzle_shapes_limit': [],
        'puzzle_is_shape_limit': [],
        'n_puzzle_limited_actions': [],
        'n_puzzle_unique_shapes': [],
        'full_user': [],
        'user': [],
        'group': [],
        'puzzle_time': [],
        'n_events_per_time': [],
        'tutorial_puzzles_rate': [],
        'n_puzzles_completed': [],
        'n_puzzle_attemps': [],
        'n_puzzle_sumbissions': [],
        'n_puzzles_dropped': [],
        'user_puzzles_success_rate': [],
        'perc_correct_shapes': [],
        'n_incorrect_shapes': [],
        'completed': []
    }
    puzzles_completed = dict()
    puzzle_tried_threshold = 30
    for full_user in data_events['full_user'].unique():
        user_features = {
            'tutorial_puzzles_complete': [],
            'n_puzzle_attemps': {},
            'n_puzzle_submissions': {},
            'time_by_puzzle': {},
            'n_puzzles_dropped': {},
        }
        user_events = data_events[data_events['full_user'] == full_user]
        user_events = user_events.sort_values(['session_id', 'time'])
        puzzles_completed[full_user] = []
        active_puzzle = None
        active_puzzle_dificulty = 0
        last_time_snap = 0.0
        time_puzzle_started = 0
        previous_event = None
        current_puzzle_tried = False
        num_events_since_last_snap = 0
        event_list = []
        undo_events = []
        time_snapshot = 0
        shape_created = False
        board_status = {}
        backup_shapes = {}
        current_level_data = {}
        for index, event in user_events.iterrows():
            event_data = json.loads(event['data'])
            if event['type'] == 'ws-start_level':
                previous_event = event
                active_puzzle = event_data['task_id']
                if active_puzzle == 'Sandbox' or active_puzzle in puzzles_completed[full_user]:
                    continue

                if active_puzzle in puzzles_solutions_files:
                    with open(f'{puzzles_solutions_folder}/{puzzles_solutions_files[active_puzzle]}') as file:
                        puzzle_solution = json.load(file)
                else:
                    puzzle_solution = None
                active_puzzle_dificulty = puzzle_set_id_to_num[json.loads(event['data'])['set_id']]
                last_time_snap = event['time']
                active_puzzle_data = df_puzzle_data[df_puzzle_data['puzzle'] == active_puzzle]
                percentile = 0.7
                time_snapshot = np.percentile(active_puzzle_data['time'], percentile) \
                    if len(active_puzzle_data) > 0 \
                    else 60

                puzzle_conditions = eval(event_data['conditions'].replace("false", "False").replace("true", "True"))
                if len(puzzle_conditions['shapeLimits']) == 0:
                    shape_limit_number = 0
                else:
                    shape_limit_number = puzzle_conditions['shapeLimits'][0] \
                        if puzzle_conditions['shapeLimits'][0] > 0 \
                        else sum(num for num in puzzle_conditions['shapeLimits'] if num > -1)
                current_level_data['n_shapes_limit'] = shape_limit_number
                current_level_data['is_limit'] = shape_limit_number > 0 or any(
                    num == 0 for num in puzzle_conditions['shapeLimits'])
                current_level_data['n_unique_shapes_limit'] = 0 if shape_limit_number == 0 else \
                    sum(1 for num in puzzle_conditions['shapeLimits'][:1] if num == -1 or num > 0)

                if puzzle_conditions['allowScale'] and puzzle_conditions['allowRotate']:
                    current_level_data['n_limited_actions'] = 2
                elif puzzle_conditions['allowScale'] or puzzle_conditions['allowRotate']:
                    current_level_data['n_limited_actions'] = 1
                else:
                    current_level_data['n_limited_actions'] = 0

                num_events_since_last_snap = 0
                if active_puzzle not in user_features['n_puzzle_attemps']:
                    user_features['n_puzzle_attemps'][active_puzzle] = 0
                    user_features['n_puzzle_submissions'][active_puzzle] = 0
                    user_features['time_by_puzzle'][active_puzzle] = 0
                    user_features['n_puzzles_dropped'][active_puzzle] = 0
                continue

            if active_puzzle is None or active_puzzle == 'Sandbox' or active_puzzle in puzzles_completed[full_user]:
                continue

            if event['type'] in list_action_events:
                num_events_since_last_snap += 1

            if previous_event is None:
                previous_event = event

            if event['type'] == 'ws-puzzle_started':
                event_list.clear()
                board_status.clear()
                backup_shapes.clear()
                current_puzzle_tried = False
                shape_created = False
                time_puzzle_started = event['time']

            if event['type'] == 'ws-create_shape':
                shape_created = True

            # PROCESS BOARD STATUS

            if event['type'] == 'ws-create_shape':
                new_shape = {
                    'shape_type': event_data['shapeType'],
                    'position': event_data['spawnPosition'],
                    'scale': {'x': 1, 'y': 1, 'z': 1},
                    'rotation': {'x': 0, 'y': 0, 'z': 0}
                }
                board_status[event_data['objSerialization']] = new_shape

            elif event['type'] == 'ws-delete_shape':
                for key in event_data['deletedShapes']:
                    backup_shapes[key] = board_status[key]
                    del board_status[key]

            elif event['type'] == 'ws-move_shape':
                if event_data['validMove']:
                    for key, value in event_data['targetOffset'].items():
                        if value == 0:
                            continue
                        for shape in event_data['selectedObjects']:
                            board_status[shape]['position'][key] += value

            elif event['type'] == 'ws-scale_shape':
                board_status[event_data['selectedObject']]['scale'] = event_data['newScale']

            elif event['type'] == 'ws-rotate_shape':
                for key, value in event_data['rotationOffset'].items():
                    if value == 0:
                        continue
                    board_status[event_data['selectedObject']]['rotation'][key] += value

            elif event['type'] == 'ws-undo_action':
                counter = -1
                last_event = event_list[counter]
                last_event_data = json.loads(last_event['data'])
                while last_event['type'] in events_not_affected_by_undo \
                        or (last_event['type'] == 'ws-move_shape' and not last_event_data['validMove']):
                    counter -= 1
                    last_event = event_list[counter]
                    last_event_data = json.loads(last_event['data'])

                if last_event['type'] in ['ws-create_shape', 'ws-delete_shape', 'ws-rotate_shape', 'ws-scale_shape',
                                          'ws-move_shape']:
                    keys = get_shape_id_from_event(last_event)

                    if last_event['type'] == 'ws-create_shape':
                        del board_status[keys[0]]

                    elif last_event['type'] == 'ws-delete_shape':
                        for key in keys:
                            board_status[key] = backup_shapes[key]

                    elif last_event['type'] == 'ws-rotate_shape':
                        for key, value in last_event_data['rotationOffset'].items():
                            if value == 0:
                                continue
                            board_status[keys[0]]['rotation'][key] -= value

                    elif last_event['type'] == 'ws-scale_shape':
                        board_status[keys[0]]['scale'] = last_event_data['newScale']

                    elif last_event['type'] == 'ws-move_shape':
                        for key, value in last_event_data['targetOffset'].items():
                            if value == 0:
                                continue
                            for shape in keys:
                                board_status[shape]['position'][key] -= value

                undo_events.append(last_event)
                event_list.pop(counter)

            elif event['type'] == 'ws-redo_action':
                last_undo_event = undo_events[-1]
                last_undo_event_data = json.loads(last_undo_event['data'])
                if last_undo_event['type'] in ['ws-create_shape', 'ws-delete_shape', 'ws-rotate_shape',
                                               'ws-scale_shape', 'ws-move_shape']:
                    if last_undo_event['type'] == 'ws-create_shape':
                        new_shape = {
                            'shape_type': last_undo_event_data['shapeType'],
                            'position': last_undo_event_data['spawnPosition'],
                            'scale': {'x': 1, 'y': 1, 'z': 1},
                            'rotation': {'x': 0, 'y': 0, 'z': 0}
                        }
                        board_status[last_undo_event_data['objSerialization']] = new_shape

                    elif last_undo_event['type'] == 'ws-delete_shape':
                        for key in last_undo_event_data['deletedShapes']:
                            del board_status[key]

                    elif last_undo_event['type'] == 'ws-rotate_shape':
                        for key, value in last_undo_event_data['rotationOffset'].items():
                            if value == 0:
                                continue
                            board_status[last_undo_event_data['selectedObject']]['rotation'][key] += value

                    elif last_undo_event['type'] == 'ws-scale_shape':
                        board_status[last_undo_event_data['selectedObject']]['scale'] = last_undo_event_data['newScale']

                    elif event['type'] == 'ws-move_shape':
                        for key, value in last_undo_event_data['targetOffset'].items():
                            if value == 0:
                                continue
                            for shape in last_undo_event_data['selectedObjects']:
                                board_status[shape]['position'][key] += value

                event_list.append(last_undo_event)
                undo_events.pop(-1)

            # END PROCESS BOARD STATUS

            # we consider that the user has tried the puzzle if he has spent a certain amount of time,
            # and they have created a shape
            if (not current_puzzle_tried) \
                    and shape_created \
                    and ((event['time'] - time_puzzle_started).seconds > puzzle_tried_threshold):
                current_puzzle_tried = True
                user_features['n_puzzle_attemps'][active_puzzle] += 1

            if event['type'] == 'ws-exit_to_menu' or event['type'] == 'ws-disconnect':
                if event['type'] == 'ws-exit_to_menu' and current_puzzle_tried:
                    user_features['n_puzzles_dropped'][active_puzzle] += 1
                active_puzzle = None
                continue

            if event['type'] == 'ws-check_solution':
                user_features['n_puzzle_attemps'][active_puzzle] += 1

            if event['type'] == 'ws-puzzle_complete':
                if event_data['task_id'] not in user_features['tutorial_puzzles_complete'] \
                        and event_data['task_id'] in tutorial_puzzles:
                    user_features['tutorial_puzzles_complete'].append(event_data['task_id'])

                if event_data['task_id'] not in puzzles_completed[full_user]:
                    puzzles_completed[full_user].append(event_data['task_id'])

                seconds_elapsed = (event['time'] - previous_event['time']).total_seconds()
                user_features['time_by_puzzle'][active_puzzle] += seconds_elapsed

                active_puzzle = None
                continue

            secs_since_last_snap = (event['time'] - last_time_snap).total_seconds()
            if active_puzzle is not None \
                    and event['type'] in player_puzzle_events \
                    and secs_since_last_snap > time_snapshot:
                features['puzzle'].append(active_puzzle)
                features['puzzle_difficulty'].append(active_puzzle_dificulty)
                features['n_puzzle_shapes_limit'].append(current_level_data['n_shapes_limit']),
                features['puzzle_is_shape_limit'].append(1 if current_level_data['is_limit'] else 0),
                features['n_puzzle_limited_actions'].append(current_level_data['n_limited_actions']),
                features['n_puzzle_unique_shapes'].append(current_level_data['n_unique_shapes_limit']),
                features['full_user'].append(full_user)
                features['user'].append(full_user.split('~')[1])
                features['group'].append(full_user.split('~')[0])
                features['puzzle_time'].append(user_features['time_by_puzzle'][active_puzzle])
                features['n_events_per_time'].append(num_events_since_last_snap / secs_since_last_snap)
                features['n_puzzle_sumbissions'].append(user_features['n_puzzle_submissions'][active_puzzle])
                features['n_puzzle_attemps'].append(user_features['n_puzzle_attemps'][active_puzzle])
                features['n_puzzles_dropped'].append(sum(user_features['n_puzzles_dropped'].values()))
                features['tutorial_puzzles_rate'].append(len(user_features['tutorial_puzzles_complete']) / 9)
                features['n_puzzles_completed'].append(len(puzzles_completed[full_user]))

                # attemp features
                if puzzle_solution is not None:
                    shape_types_in_board = [shape['shape_type'] for shape in board_status.values()]
                    shape_types_in_solution = [shape['shapeType'] for shape in puzzle_solution['shapeData']]
                    n_correct_shapes = 0
                    n_incorrect_shapes = 0

                    for shape in shape_types_in_board:
                        if shape in shape_types_in_solution:
                            n_correct_shapes += 1
                            shape_types_in_solution.remove(shape)
                        else:
                            n_incorrect_shapes += 1

                    features['perc_correct_shapes'].append(n_correct_shapes / len(puzzle_solution['shapeData']))
                    features['n_incorrect_shapes'].append(n_incorrect_shapes)
                else:
                    features['perc_correct_shapes'].append(0)
                    features['n_incorrect_shapes'].append(0)

                puzzles_tried = sum(
                    1 for k, v in user_features['n_puzzle_attemps'].items() if k != active_puzzle and v > 0)
                features['user_puzzles_success_rate'].append(
                    1 if puzzles_tried == 0 else len(puzzles_completed[full_user]) / puzzles_tried
                )

                features['completed'].append(False)

                last_time_snap = event['time']
                time_snapshot = 40
                num_events_since_last_snap = 0

            # we consider that the user is active if the last action was done in the last user_active_threshold seconds
            seconds_elapsed = (event['time'] - previous_event['time']).total_seconds()
            if active_puzzle is not None and seconds_elapsed < user_active_threshold:
                user_features['time_by_puzzle'][active_puzzle] += seconds_elapsed

            previous_event = event

            if event['type'] not in ['ws-undo_action', 'ws-redo_action']:
                event_list.append(event)

    features_df = pd.DataFrame.from_dict(features, orient='columns')
    for full_user in puzzles_completed:
        filt = (features_df['full_user'] == full_user) & (features_df['puzzle'].isin(puzzles_completed[full_user]))
        features_df.loc[filt, 'completed'] = True

    return features_df


def get_shape_id_from_event(event):
    event_data = json.loads(event['data'])
    if event['type'] == 'ws-create_shape':
        return [event_data['objSerialization']]

    if event['type'] == 'ws-delete_shape':
        return event_data['deletedShapes']

    if event['type'] == 'ws-rotate_shape' or event['type'] == 'ws-scale_shape':
        return [event_data['selectedObject']]

    if event['type'] == 'ws-move_shape':
        return event_data['selectedObjects']


data_loaded = load_data(data_path)

df_puzzle_data = extract_puzzle_features(data_loaded)
df_puzzle_data.to_csv(puzzle_data_path_export)

extracted_features = extract_feautres(data_loaded)
extracted_features.to_csv(features_path_export)