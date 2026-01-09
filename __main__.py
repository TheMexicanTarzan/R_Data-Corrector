from pathlib import Path
import logging
import json

from src import (
    read_csv_files_to_polars,
    run_full_sanity_check,
    run_full_statistical_filter,
    run_full_pipeline,
    launch_dashboard
    )
current_dir = Path.cwd()
data_directory = current_dir / "Input" / "Data"
metadata_path = current_dir / "Input" / "Universe_Information" / "Universe_Information.csv"
output_logs_directory = current_dir / "Output"
out_format = "csv"
batch_size = 512
max_files = 5
save_data = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


dataframe_dict = read_csv_files_to_polars(data_directory, metadata_path=metadata_path, max_files=max_files)

original_file_paths = {
    ticker: data_directory / ticker for ticker in dataframe_dict.keys()
}

# clean_data_dict, logs = run_full_sanity_check(dataframe_dict,
#                                               save_data=save_data,
#                                               out_format=out_format,
#                                               output_logs_directory=output_logs_directory,
#                                               batch_size=batch_size)

# clean_data_dict,logs = run_full_statistical_filter(dataframe_dict,
#                                                    save_data=save_data,
#                                                    out_format=out_format,
#                                                    output_logs_directory=output_logs_directory,
#                                                    batch_size=batch_size)

# clean_data_dict, logs_sanity, logs_stats = run_full_pipeline(dataframe_dict,
#                                           save_data=save_data,
#                                           out_format=out_format,
#                                           output_logs_directory=output_logs_directory,
#                                           batch_size=batch_size)

launch_dashboard()