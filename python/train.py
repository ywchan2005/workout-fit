import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def read_data(path):
    df = pd.read_csv(path)
    x_df = df.pivot(index=['pose_id', 'frame_id'], columns='keypoint_id', values='x')
    y_df = df.pivot(index=['pose_id', 'frame_id'], columns='keypoint_id', values='y')
    z_df = df.pivot(index=['pose_id', 'frame_id'], columns='keypoint_id', values='z')
    pose_df = x_df.merge(y_df, left_index=True, right_index=True, suffixes=('_x', '_y'))
    pose_df = pose_df.merge(z_df.rename(columns=lambda c: f'{c}_z'), left_index=True, right_index=True)
    pose_df = pose_df.reset_index(drop=False)
    return pose_df

def unify_data(df):
    ### 0
    pose0_df = pd.concat([
        df[df.pose_id == 4],
        df[df.pose_id == 8],
    ], axis=0).reset_index(drop=True)
    pose0_df.pose_id = 0
    ### 1
    pose1_df = pd.concat([
        df[df.pose_id == 1],
        df[df.pose_id == 3],
        df[df.pose_id == 5],
        df[df.pose_id == 7],
    ], axis=0).reset_index(drop=True)
    pose1_df.pose_id = 1
    ### 2
    pose2_df = pd.concat([
        df[df.pose_id == 2],
        df[df.pose_id == 6],
    ], axis=0).reset_index(drop=True)
    pose2_df.pose_id = 2
    df = pd.concat([
        pose0_df,
        pose1_df,
        pose2_df,
    ], axis=0).reset_index(drop=True)
    return df

def interpolate_data(df, n_copies=2):
    df = pd.concat([
        df,
        *[__interpolate_data(df, i) for i in range(
            df.pose_id.min(),
            df.pose_id.max(),
        ) for _ in range(n_copies)],
    ], axis=0).reset_index(drop=True)
    return df

def __interpolate_data(df, n):
    pose2_df = df[df.pose_id == n + 1].reset_index(drop=True)
    pose1_df = df[df.pose_id == n].reset_index(drop=True)
    chosen = np.random.choice(pose1_df.index, len(pose2_df))
    pose1_df = pose1_df.iloc[chosen].reset_index(drop=True)
    r = np.random.uniform(0, 1, len(pose1_df))
    interpolated_df = pose1_df.multiply(r, axis=0) + pose2_df.multiply(1 - r, axis=0)
    return interpolated_df

def main():
    print(f'xgboost: {xgb.__version__}')
    print(f'pandas: {pd.__version__}')

    pose_df = pd.concat([read_data(path) for path in [
        'dataset/pose_2.csv',
        'dataset/pose_3.csv',
        'dataset/pose_4.csv',
        'dataset/pose_6.csv',
        'dataset/pose_7.csv',
        'dataset/pose_8.csv',
        'dataset/pose_9.csv',
        'dataset/pose_10.csv',

        'dataset/pose_11.csv',
        'dataset/pose_12.csv',
        'dataset/pose_13.csv',
        'dataset/pose_14.csv',
        'dataset/pose_15.csv',
        'dataset/pose_16.csv',
        'dataset/pose_18.csv',
        'dataset/pose_19.csv',

        'dataset/pose_20.csv',
        'dataset/pose_21.csv',
        'dataset/pose_22.csv',
        'dataset/pose_23.csv',
        'dataset/pose_24.csv',
        'dataset/pose_25.csv',
        'dataset/pose_26.csv',
        'dataset/pose_27.csv',
    ]], axis=0).reset_index(drop=True)
    print(pose_df.shape)
    print(pose_df.head())
    print('-' * 50)

    train_df, test_df = train_test_split(
        pose_df,
        test_size=0.3,
        random_state=42
    )
    print(f'train: {train_df.shape}')
    print(f'test: {test_df.shape}')

    train_df = unify_data(train_df)
    train_df = interpolate_data(train_df, n_copies=16)

    test_df = unify_data(test_df)
    test_df = interpolate_data(test_df, n_copies=16)

    m = xgb.XGBRegressor()
    m.fit(
        train_df.drop(['pose_id', 'frame_id'], axis=1).to_numpy(),
        train_df.pose_id.to_numpy(),
    )
    predicted_pose_id = m.predict(
        test_df.drop(['pose_id', 'frame_id'], axis=1).to_numpy(),
    )
    m.save_model('model.json')

    result_df = pd.DataFrame({
        'target': test_df.pose_id,
        'prediction': predicted_pose_id,
    })
    result_df['abs diff'] = np.abs(result_df.target - result_df.prediction)
    result_df = result_df.sort_values('abs diff', ascending=False)
    print(result_df.head())
    print((result_df['abs diff'] ** 2).mean() ** .5)

    sns.displot(result_df['abs diff']).figure.savefig('diff.png')

    result_df = pd.DataFrame({
        'target': ((test_df.pose_id * 2).round().abs() * .5).apply(lambda x: f'{x}'),
        'prediction': [f'{x}' for x in np.abs((predicted_pose_id * 2).round() * .5)],
    })
    labels = sorted(result_df.target.unique())
    sns.heatmap(confusion_matrix(
        result_df.target,
        result_df.prediction,
        labels=labels,
    ), xticklabels=labels, yticklabels=labels, annot=True, cbar=False, fmt=".0f").figure.savefig('confusion.png')
    print()
    print(classification_report(
        result_df.target,
        result_df.prediction,
    ))

    importance = m.get_booster().get_score(importance_type='weight')
    importance = pd.DataFrame.from_dict(importance, orient='index')
    print(importance.sort_values(0, ascending=False).head(5))

if 'main' in __name__:
    main()