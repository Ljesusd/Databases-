# Set the logging level of logguru to INFO
import sys
import pathlib as path
from loguru import logger
import numpy as np
import pandas as pd

sys.path.append('../common_eurobench')
sys.path.append('/Users/jorge/Documents/Coding/Projects/nrg-standardization/common_eurobench')

import exploration.exploration_utils as explore
import characterization.characterization_utils as char

from conversion.convert_utils import DataClass
import conversion.convert_utils as conv


def eliminate_general_turns(df_preprocess, file, info, events):
    try:
        # Remove and split
        indexes = conv.find_indices_remove_nullify_divide_using_tuple(df_preprocess, events,
                                                                      ('general_start_turn', 'general_end_turn'),
                                                                      tol=0.1)

        # Split the dataframes using the events to create a list of dataframes
        df_preprocess.loc[indexes, :] = np.nan
        list_new_df = conv.divide_dataframe(df_preprocess)

        # For each one of the events in the dictionary events, revise the events to check that they are within the limits
        # of the time column for each dataframe
        list_new_events = []
        for curr_df in list_new_df:
            max_time = curr_df['time'].iloc[-1]
            min_time = curr_df['time'].iloc[0]
            new_events = {}
            for key, val in events.items():
                new_events[key] = [x for x in val if min_time < x < max_time]

            list_new_events.append(new_events)

        return list_new_df, file, info, list_new_events
    except Exception as e:
        logger.error(f"Error in eliminate_general_turns: {e}")
        return None


def process_file(eliminate_general_turns, route_root_experiments, subject, session, condition, experiment_out,
                 results_path, n_samples, var_order):
    route_eurobench_data = path.Path(route_root_experiments).joinpath(experiment_out, subject, session,
                                                                      'IRREGULAR_TERRAIN')
    print(f'Route Eurobench data: {route_eurobench_data}')

    # Standard route for VICON data
    route_out_vicon = path.Path(route_eurobench_data).joinpath('VICON')
    route_out_vicon.mkdir(parents=True, exist_ok=True)
    print(f'Route Vicon OUT: {route_out_vicon}')

    # Standard route for IMU data
    route_out_xsens = path.Path(route_eurobench_data).joinpath('XSENS')
    route_out_xsens.mkdir(parents=True, exist_ok=True)
    print(f'Route Delsys OUT: {route_out_xsens}')

    # Out route for the figures
    route_results = path.Path(results_path).joinpath('RESULTS').joinpath(subject)
    route_figures_individual = route_results.joinpath('figures_xsens_resampled').joinpath('condition_' + condition)
    route_figures_average = route_results.joinpath('figures_xsens_resampled_average').joinpath('condition_' + condition)

    # create a glob pattern to find all the files that match the pattern subject_\d+_cond_01_run_\d+_jointAngles.csv
    pattern = f'**/subject_*_cond_{condition}_run_*_jointAngles.csv'
    files = list(path.Path(route_out_xsens).glob(pattern))
    print(files)

    # NOTE: This is the path where the events given by the Vicon system are located. If there are no events,
    # then this should be None
    path_events = route_out_vicon

    # Resample left
    split_events_category = 'l_heel_strike'
    pattern_files = r'(subject_\d+_cond_\d+_run_\d+)'
    remove_events_category = ('general_start_turn', 'general_end_turn')
    data_class_left = conv.convert_dir_eurobench_to_dataclass(files, columns=None,
                                                              path_events=path_events, pattern=pattern_files,
                                                              split_events_category=split_events_category,
                                                              split_using_events=True,
                                                              remove_nullify_divide_using_events='divide',
                                                              remove_nullify_divide_events_category=remove_events_category)
    dfs_resampled_left = explore.resample_dataclass(data_class_left[0], 'n_samples', n_samples)

    # Resample right
    split_events_category = 'r_heel_strike'
    pattern_files = r'(subject_\d+_cond_\d+_run_\d+)'
    remove_events_category = ('general_start_turn', 'general_end_turn')
    data_class_right = conv.convert_dir_eurobench_to_dataclass(files, columns=None,
                                                               path_events=path_events, pattern=pattern_files,
                                                               split_events_category=split_events_category,
                                                               split_using_events=True,
                                                               remove_nullify_divide_using_events='divide',
                                                               remove_nullify_divide_events_category=remove_events_category)
    dfs_resampled_right = explore.resample_dataclass(data_class_right[0], 'n_samples', n_samples)

    # Combine left and right
    dfs_resampled_all = DataClass(inner_nest_by=dfs_resampled_right.inner_nest_by,
                                  outer_nest_by=dfs_resampled_right.outer_nest_by)
    # Extract from dfs_resampled_right only the keys that do not start with l_ -> That means that they are right or
    # nonlateral
    dfs_resampled_all.data_class = {key: dfs_resampled_right.data_class[key]
                                    for key in dfs_resampled_right.data_class.keys() if not key.startswith('l_')}
    # Update dfs_resampled_all with the keys that start with l_
    dfs_resampled_all.data_class.update({key: dfs_resampled_left.data_class[key]
                                         for key in dfs_resampled_left.data_class.keys() if key.startswith('l_')})

    # Reorder dfs_resampled_all.data_class according to the values of var_order
    dfs_resampled_all.data_class = {key: dfs_resampled_all.data_class[key] for key in var_order}

    # Individual plots
    explore.plot_dataclass_individually(dfs_resampled_all, bool_show=False, bool_save=True,
                                        route_figures=route_figures_individual, col_time=None, plot_events=False)

    # Average plots
    explore.plot_dataclass_average(dfs_resampled_all, bool_show=False, bool_std=True,
                                   save_directory=route_figures_average)

    # Data ala URJC. According to URJC requirements
    dataframes_averages = []
    dataframes_all = {}
    for key in dfs_resampled_all.data_class.keys():
        logger.info(f"key: {key}")

        if key == 'time':
            continue

        for key_inner, val_inner in dfs_resampled_all.data_class[key].outer_nest.items():
            logger.info(f"key_inner: {key_inner}")

            if key_inner not in dataframes_all:
                dataframes_all[key_inner] = {}
            if key not in dataframes_all[key_inner]:
                dataframes_all[key_inner][key] = []
            # Append the data to the list
            dataframes_all[key_inner][key].extend(val_inner.data)

        # Average values is a list, create a dataframe from it
        average_values = dfs_resampled_all.data_class[key].statistics_over_inner_nest['average']
        dataframes_averages.append(pd.DataFrame(average_values, columns=[key]))

    # Concatenate all the dataframes in the list
    df_average = pd.concat(dataframes_averages, axis=1)
    # Reorder the columns, the elements of var_order first, then the rest
    df_average = df_average[var_order + [col for col in df_average.columns if col not in var_order]]

    # Save the dataframe in a xls file
    route_data_average = path.Path(route_results).joinpath('data_average')
    path.Path.mkdir(route_data_average, parents=True, exist_ok=True)
    df_average.to_excel(route_data_average.
                        joinpath(f'subject_{subject}_{session}_condition_{condition}.xlsx'))

    # Save in a xls file a sheet for each key_inner
    route_data_individual_xls = path.Path(route_results).joinpath('data_individual_xls')

    path.Path.mkdir(route_data_individual_xls, parents=True, exist_ok=True)

    for key, val_all in dataframes_all.items():
        full_route = route_data_individual_xls.joinpath(
            f'{key[:-4]}.xlsx')
        print(full_route)
        with pd.ExcelWriter(full_route) as writer:
            print(key)
            for key_inner, val_inner in val_all.items():
                print(key_inner)
                column_names = [key] * len(val_inner)
                df = pd.DataFrame(np.array(val_inner).T)
                df.to_excel(writer, sheet_name=key_inner, index=False, header=False)

    # # Save in a separate xls file each key_inner
    # route_data_individual = path.Path(route_results).joinpath('data_individual')
    # path.Path.mkdir(route_data_individual, parents=True, exist_ok=True)
    # for key, val_all in dataframes_all.items():
    #     for key_inner, val_inner in val_all.items():
    #         df = pd.DataFrame(np.array(val_inner).T, columns=[key] * len(val_inner))
    #         filename = f'resampled_individual_{subject}_{session}_condition_{condition}_{key_inner}.xlsx'
    #         df.to_excel(route_data_individual.joinpath(filename))

    # Patterns for the characterization
    pattern_subject_condition_run = r'subject_(?P<subject>\w+)_cond_(?P<condition>\w+)_run_(?P<run>\w+)_Trajectories.csv'
    pattern_info = r'(subject_\d+_cond_\d+_run_\d+)'
    pattern_event = r'(subject_\d+_cond_\d+_run_\d+)'

    pattern_glob_csv = f'**/subject_*_cond_{condition}_run_*_Trajectories.csv'

    # Characterize all files in route_in
    route_out_dir_csv = path.Path(route_results).joinpath('csv_spatiotemporal_features')
    char.characterize_dir(route_out_xsens, route_out_dir_csv, pattern_glob_csv,
                          function_preprocess_in=eliminate_general_turns,
                          function_characterize_in='calculate_spatiotemporal_features_from_dataframe',
                          pattern_subject_condition_run=pattern_subject_condition_run,
                          pattern_info=pattern_info, pattern_event=pattern_event, route_yaml=path_events)


if __name__ == '__main__':

    # This route has to be changed depending on the subject to characterize_element
    route_root_experiments = '/Users/jorge/BasesDatos/NEUROMARK/'
    # Orden requerido por URJC
    var_order = ['l_elbow_flexion', 'r_elbow_flexion', 'l_elbow_pronation', 'r_elbow_pronation', 'l_elbow_deviation',
                 'r_elbow_deviation',
                 'l_knee_flexion', 'r_knee_flexion', 'l_knee_adduction', 'r_knee_adduction', 'l_knee_rotation',
                 'r_knee_rotation',
                 'l_ankle_flexion', 'r_ankle_flexion', 'l_ankle_adduction', 'r_ankle_adduction', 'l_ankle_rotation',
                 'r_ankle_rotation',
                 'l_hip_rotation', 'r_hip_rotation', 'l_hip_adduction', 'r_hip_adduction', 'l_hip_flexion',
                 'r_hip_flexion',
                 'l_wrist_flexion', 'r_wrist_flexion', 'l_wrist_pronation', 'r_wrist_pronation', 'l_wrist_deviation',
                 'r_wrist_deviation',
                 'l_shoulder_flexion', 'r_shoulder_flexion', 'l_shoulder_adduction', 'r_shoulder_adduction',
                 'l_shoulder_rotation', 'r_shoulder_rotation',
                 'l_t4_shoulder_flexion', 'r_t4_shoulder_flexion', 'l_t4_shoulder_adduction', 'r_t4_shoulder_adduction',
                 'l_t4_shoulder_rotation', 'r_t4_shoulder_rotation',
                 'l_ballfoot_flexion', 'r_ballfoot_flexion', 'l_ballfoot_adduction', 'r_ballfoot_adduction',
                 'l_ballfoot_rotation', 'r_ballfoot_rotation',
                 'l5s1_flexion', 'l5s1_lateralFlexion', 'l5s1_axialFlexion',
                 'l4l3_flexion', 'l4l3_lateralFlexion', 'l4l3_rotation',
                 't9t8_flexion', 't9t8_lateralFlexion', 't9t8_rotation',
                 'l1t12_flexion', 'l1t12_lateralFlexion', 'l1t12_rotation',
                 't1c7_flexion', 't1c7_rotation', 't1c7_lateralFlexion',
                 'head_flexion', 'head_lateralFlexion', 'head_rotation',
                 ]

    # subject = 'SUBJECT_27'
    session = 'SESSION_01'
    # condition = '02'
    experiment_in = 'EXPORTED_DATA'
    experiment_out = 'EUROBENCH_DATA'
    results_path = path.Path.cwd()
    n_samples = 101

    list_subject = [
        '15', '16', '17', '18', '23', '24',
        '25', '26', '29', '30', '31',
        '33', '34', '35', '36', '37', '38',
        '40', '42', '43', '44', '45']  #'27','20','39','32',
    # list_subject = ['39'] -> run_04: da error
    # list_condition = ['03'] -> run_04: da error
    # Processing file subject_32_cond_05_run_04_jointAngles.csv da error, el mismo error para subject_45_cond_05_run_04
    # 'Error finding the indices to remove, nullify or divide using the tuple: Initial event 9.300000190734863 is greater than fina...

    list_condition = ['01', '02', '03', '04', '05']  #

    # Subject 05 -> sin datos XSens
    for curr_condition in list_condition:
        for subject in list_subject:
            curr_subject = f'SUBJECT_{subject}'
            process_file(eliminate_general_turns, route_root_experiments, curr_subject, session, curr_condition,
                         experiment_out,
                         results_path, n_samples, var_order)
