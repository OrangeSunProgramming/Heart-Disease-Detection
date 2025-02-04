import cnn_model
import cnn_model as cnnm
from numpy import average
import optuna
import optuna.visualization as vis
import tensorflow as tf
import data_preparation as dp
import sample_weights as sw
import inspect
from custom_f1_score import F1Score
from custom_hamming_loss import HammingLoss
from optuna.pruners import SuccessiveHalvingPruner
from tensorflow.keras.optimizers import Adam
from optuna.samplers import NSGAIISampler



#Optuna hyperparameter tuning with TPE
class OptunaTPE:
    def __init__(self, model, metric, active=False):
        self.active = active
        self.model = model
        self.metric = metric
        self.study = None

    def objective(self, trial):
        signature = inspect.signature(self.model)

        kwargs = {}
        for param in signature.parameters.values():
          if param.name == "input_shape":
            continue
        
          if "conv" in param.name:
            kwargs[param.name] = trial.suggest_int(param.name, 64, 640, step=32)
          elif "drop" in param.name:
            kwargs[param.name] = trial.suggest_float(param.name, 0.0, 0.5, step=0.1)
          elif "dense" in param.name:
            kwargs[param.name] = trial.suggest_int(param.name, 128, 640, step=32)

        learning_rate = trial.suggest_categorical('learning_rate', [2e-4, 1e-4, 1e-3, 1e-2, 3e-4])
        model_set = self.model(input_shape=(128, 128, 1), **kwargs)

        # Compile model
        model_set.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=[
                          F1Score(num_classes=5, average='macro'),
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall'),
                          tf.keras.metrics.AUC(multi_label=True, num_labels=5, name='auc'),
                          HammingLoss()
                      ])

        # Train the model for 10 epochs
        for epoch in range(10):
            history = model_set.fit(dp.X_train, dp.y_train, epochs=1, validation_data=(dp.X_val, dp.y_val),
                                verbose=0, batch_size=dp.batch_size, sample_weight=sw.sample_weights_train)
            val_metric = history.history[str(self.metric)][0]

            # Report intermediate objective value
            trial.report(val_metric, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_metric

    def run_optimization(self, direction, n_trials):
        if self.active:
            self.study = optuna.create_study(direction=direction, pruner=SuccessiveHalvingPruner())
            self.study.optimize(self.objective, n_trials=n_trials)

    def optimization_history(self):
        if self.study:
            history_plot = vis.plot_optimization_history(self.study)
            return history_plot.show()

    def hyperparameter_importance(self):
        if self.study:
            importance_plot = vis.plot_param_importances(self.study)
            return importance_plot.show()

    def parallel_coordinate_plot(self):
        if self.study:
            parallel_plot = vis.plot_parallel_coordinate(self.study)
            return parallel_plot.show()

    def contour_plot(self):
        if self.study:
            contour_plot_vis = vis.plot_contour(self.study)
            return contour_plot_vis.show()


#Optuna with NSGAIISampler
class OptunaNSGAIISampler:
    def __init__(self, model, metric1, metric2, active=False):
        self.active = active
        self.model = model
        self.metric1 = metric1
        self.metric2 = metric2
        self.study = None

    def objective(self, trial):
        signature = inspect.signature(self.model)

        kwargs = {}
        for param in signature.parameters.values():
          if param.name == "input_shape":
            continue
          
          if "conv" in param.name:
            kwargs[param.name] = trial.suggest_int(param.name, 64, 320, step=32)
          elif "drop" in param.name:
            kwargs[param.name] = trial.suggest_float(param.name, 0.0, 0.5, step=0.1)
          elif "dense" in param.name:
            kwargs[param.name] = trial.suggest_int(param.name, 64, 320, step=32)

        learning_rate = trial.suggest_categorical('learning_rate', [2e-4, 1e-4, 1e-3, 1e-2, 3e-4])
        model_set = self.model(input_shape=(128, 128, 1), **kwargs)

        #Compile the model
        model_set.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=[
                          F1Score(num_classes=5, average='macro'),
                          tf.keras.metrics.Precision(),
                          tf.keras.metrics.Recall(),
                          tf.keras.metrics.AUC(multi_label=True, num_labels=5, name='auc'),
                          HammingLoss()
                      ])

        # Train the model for 10 epochs
        for epoch in range(10):
            history = model_set.fit(dp.X_train, dp.y_train, epochs=1, validation_data=(dp.X_val, dp.y_val), verbose=0, batch_size=dp.batch_size)
            train_rep_metric1 = []
            train_rep_metric2 = []

            if "val_" in str(self.metric1):
              train_rep_metric1.append(str(self.metric1).replace("val_", ""))

            val_metric1 = history.history[str(self.metric1)][0]
            train_metric1 = history.history[train_rep_metric1[0]][0]

            if "val_" in str(self.metric2):
              train_rep_metric2.append(str(self.metric2).replace("val_", ""))

            val_metric2 = history.history[str(self.metric2)][0]
            train_metric2 = history.history[train_rep_metric2[0]][0]

            # If train and validation losses or F1 scores differ too much, prune the trial
            if abs(train_metric1 - val_metric1) > 0.2:
                raise optuna.exceptions.TrialPruned()

            if abs(train_metric2 - val_metric2) > 0.2:
                raise optuna.exceptions.TrialPruned()

        return val_metric1, val_metric2

    def run_optimization(self, directions, n_trials):
        if self.active:
            # Create study with both minimization and maximization objectives
            self.study = optuna.create_study(directions=directions,
                                             sampler=NSGAIISampler(),
                                             pruner=SuccessiveHalvingPruner())
            # Optimize the objective function
            self.study.optimize(self.objective, n_trials=n_trials)

    def optimization_history(self, target_name=None):
        if self.study:
            target_func = lambda t: t.values[0] if target_name == str(self.metric1) else t.values[1]
            history_plot = vis.plot_optimization_history(self.study, target=target_func, target_name=str(self.metric1))
            history_plot.show()

    def hyperparameter_importance(self, target_name=None):
        if self.study:
            target_func = lambda t: t.values[0] if target_name == str(self.metric1) else t.values[1]
            importance_plot = vis.plot_param_importances(self.study, target=target_func, target_name=str(self.metric1))
            importance_plot.show()

    def parallel_coordinate_plot(self, target_name=None):
        if self.study:
            target_func = lambda t: t.values[0] if target_name == str(self.metric1) else t.values[1]
            parallel_plot = vis.plot_parallel_coordinate(self.study, target=target_func, target_name=str(self.metric1))
            parallel_plot.show()

    def contour_plot(self, target_name=None):
        if self.study:
            target_func = lambda t: t.values[0] if target_name == str(self.metric1) else t.values[1]
            contour_plot = vis.plot_contour(self.study, target=target_func, target_name=str(self.metric1))
            contour_plot.show()

    def pareto_front(self):
        if self.study:
            pareto_plot = vis.plot_pareto_front(self.study, target_names=[str(self.metric1), str(self.metric1)])
            pareto_plot.show()