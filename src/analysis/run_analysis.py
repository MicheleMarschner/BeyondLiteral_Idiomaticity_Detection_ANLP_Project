import pandas as pd

from config import PATHS, Paths
from analysis.evaluate_subslices import evaluate_subslices
from analysis.create_subslices import add_ambiguous_slices, build_slices_and_ids
from analysis.stress_masking import run_stress_masking_over_all_runs
from utils.helper import copy_original_dataset, write_json



def run_analysis(split_type: str, project_paths: Paths = PATHS):

    for setting in ["one_shot", "zero_shot"]:
        data_path = project_paths.data_preprocessed / f"{setting}_splits/{setting}_{split_type}.csv"
        analysis_data_path = project_paths.data_analysis / f"{setting}_{split_type}_analysis.csv"
        slice_ids_path = project_paths.data_analysis / f"{setting}_{split_type}_slice_ids.json"

        df = pd.read_csv(data_path)

        if not analysis_data_path.exists():
            copy_original_dataset(data_path, analysis_data_path)

            df_with_slices, slice_ids = build_slices_and_ids(df, min_total=5)

            # write updated columns back to the analysis CSV
            df_with_slices.to_csv(analysis_data_path, index=False)

            add_ambiguous_slices(csv_path=analysis_data_path, hard_ids=slice_ids["ambiguous_mwe_ids"])

            # save IDs json (contains both ambiguity + freqbin slices)
            write_json(slice_ids_path, slice_ids)


    evaluate_subslices(split_type="test")
    
    run_stress_masking_over_all_runs(PATHS.runs, PATHS.results)
