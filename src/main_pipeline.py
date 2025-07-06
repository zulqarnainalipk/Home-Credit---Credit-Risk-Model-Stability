
import gc
import joblib
import pandas as pd
from pathlib import Path

from .data_preprocessing import Pipeline
from .feature_aggregation import Aggregator
from .data_loader import read_file, read_files
from .feature_engineering import feature_eng
from .utils import to_pandas, reduce_mem_usage
from .ensemble_model import VotingModel

# Define root and data directories
ROOT = Path("/kaggle/input/home-credit-credit-risk-model-stability")
TRAIN_DIR = ROOT / "parquet_files" / "train"
TEST_DIR = ROOT / "parquet_files" / "test"

def run_pipeline(mode="train"):
    if mode == "train":
        print("Running pipeline in training mode...")
        data_store = {
            "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
            "depth_0": [
                read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
                read_files(TRAIN_DIR / "train_static_0_*.parquet"),
            ],
            "depth_1": [
                read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
                read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
                read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
                read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
                read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
                read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
                read_file(TRAIN_DIR / "train_other_1.parquet", 1),
                read_file(TRAIN_DIR / "train_person_1.parquet", 1),
                read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
                read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
            ],
            "depth_2": [
                read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
            ]
        }

        df_train = feature_eng(**data_store)
        del data_store
        gc.collect()

        df_train = df_train.pipe(Pipeline.filter_cols)
        df_train, cat_cols = to_pandas(df_train)
        df_train = reduce_mem_usage(df_train)

        # Handle missing values and reduce columns based on correlation (from notebook)
        nums = df_train.select_dtypes(exclude='category').columns
        nans_df = df_train[nums].isna()
        nans_groups = {}
        for col in nums:
            cur_group = nans_df[col].sum()
            try:
                nans_groups[cur_group].append(col)
            except:
                nans_groups[cur_group]=[col]
        del nans_df; gc.collect()

        uses = []
        for k,v in nans_groups.items():
            if len(v)>1:
                Vs = nans_groups[k]
                # This function is not defined in the notebook, assuming it's a placeholder or needs to be implemented
                # For now, just taking the first column from the group as a simplification
                # grps = group_columns_by_correlation(df_train[Vs], threshold=0.8)
                # use = reduce_group(grps)
                use = [v[0]] # Simplified for now
                uses = uses + use
            else:
                uses = uses + v
        df_train = df_train[uses]

        # Placeholder for model training (as per notebook, it trains a single LGBM)
        # This part would typically be in a separate training script or module
        # fitted_models_lgb = []
        # model = lgb.LGBMClassifier()
        # model.fit(df_train, y)
        # fitted_models_lgb.append(model)

        # Save processed data for later use
        joblib.dump((df_train, None, None), 'processed_train_data.pkl') # y and df_test will be handled separately
        print("Training data processed and saved.")

    elif mode == "predict":
        print("Running pipeline in prediction mode...")
        # Load pre-trained models and processed training data info (cols, cat_cols)
        lgb_notebook_info = joblib.load('/kaggle/input/homecredit-models-public/other/lgb/1/notebook_info.joblib')
        cols = lgb_notebook_info['cols']
        cat_cols = lgb_notebook_info['cat_cols']
        lgb_models = joblib.load('/kaggle/input/homecredit-models-public/other/lgb/1/lgb_models.joblib')
        cat_notebook_info = joblib.load('/kaggle/input/homecredit-models-public/other/cat/1/notebook_info.joblib')
        cat_models = joblib.load('/kaggle/input/homecredit-models-public/other/cat/1/cat_models.joblib')

        model = VotingModel(lgb_models + cat_models, cat_cols=cat_cols)

        data_store = {
            "df_base": read_file(TEST_DIR / "test_base.parquet"),
            "depth_0": [
                read_file(TEST_DIR / "test_static_cb_0.parquet"),
                read_files(TEST_DIR / "test_static_0_*.parquet"),
            ],
            "depth_1": [
                read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
                read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
                read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
                read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
                read_files(TEST_DIR / "test_credit_bureau_a_1_*.parquet", 1),
                read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
                read_file(TEST_DIR / "test_other_1.parquet", 1),
                read_file(TEST_DIR / "test_person_1.parquet", 1),
                read_file(TEST_DIR / "test_deposit_1.parquet", 1),
                read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
            ],
            "depth_2": [
                read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
                read_files(TEST_DIR / "test_credit_bureau_a_2_*.parquet", 2),
                read_file(TEST_DIR / "test_applprev_2.parquet", 2),
                read_file(TEST_DIR / "test_person_2.parquet", 2)
            ]
        }

        df_test = feature_eng(**data_store)
        del data_store
        gc.collect()

        df_test = df_test.select(['case_id'] + cols)
        df_test, _ = to_pandas(df_test, cat_cols)
        df_test = reduce_mem_usage(df_test)
        df_test = df_test.set_index('case_id')
        gc.collect()

        y_pred = pd.Series(model.predict_proba(df_test)[:, 1], index=df_test.index)
        condition = y_pred < 0.98
        df_subm = pd.read_csv(ROOT / "sample_submission.csv")
        df_subm = df_subm.set_index("case_id")
        df_subm["score"] = y_pred
        df_subm.loc[condition, 'score'] = (df_subm.loc[condition, 'score'] - 0.073).clip(0)
        df_subm.to_csv("submission.csv")
        print("Prediction completed and submission file generated.")

if __name__ == "__main__":
    # Example usage: run_pipeline("train") or run_pipeline("predict")
    # For this task, we'll just create the file structure and modules.
    pass


