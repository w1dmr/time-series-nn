import torch
import numpy as np


def test_model(d_train, d_test, test_data, model, device):
    out_dim = d_test.out_features
    mae_vals = []
    smape_vals = []
    mse_vals = []

    model.eval()

    for i, (x_test, y_test) in enumerate(test_data):
        with torch.no_grad():
            x_test, y_test = x_test.to(device), y_test.to(device)
            # predict = model(x_test)
            predict = model(x_test).squeeze(0)

            if d_test.normalize_targets:
                y_test = y_test * torch.tensor(d_train.y_std, device=device) + torch.tensor(d_train.y_mean,
                                                                                            device=device)
                predict = predict * torch.tensor(d_train.y_std, device=device) + torch.tensor(d_train.y_mean,
                                                                                              device=device)

            # if i % 365 == 0:
            #     print(f"Target: {y_test.cpu().numpy()}")
            #     print(f"Predict: {predict.cpu().numpy()}")

            mae_vals.append(torch.abs(predict - y_test).cpu().numpy())  # MAE отдельно по всем параметрам
            smape_vals.append((100 * torch.abs(predict - y_test) / (
                    torch.abs(predict) + torch.abs(
                y_test))).cpu().numpy())  # SMAPE отдельно по всем параметрам
            mse_vals.append(((predict - y_test) ** 2).cpu().numpy())  # MSE отдельно по всем параметрам

    mae_vals = np.concatenate(mae_vals, axis=0)
    smape_vals = np.concatenate(smape_vals, axis=0)
    mse_vals = np.concatenate(mse_vals, axis=0)
    rmse_vals = np.sqrt(mse_vals)

    mean_mae_per_feature = np.mean(mae_vals, axis=0)
    mean_smape_per_feature = np.mean(smape_vals, axis=0)
    mean_mse_per_feature = np.mean(mse_vals, axis=0)
    mean_rmse_per_feature = np.mean(rmse_vals, axis=0)

    return mean_mae_per_feature, mean_smape_per_feature
