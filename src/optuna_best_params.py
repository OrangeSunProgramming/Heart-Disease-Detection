from HyperPtuning import OptunaTPE, OptunaNSGAIISampler

#Defining Error Classes
class MultipleTrueError(Exception):
  def __init__(self, message):
    self.message = message
    super().__init__(self.message)

class TrialSelectionError(Exception):
  def __init__(self, message):
    self.message = message
    super().__init__(self.message)

class TrialChoiceError(Exception):
  def __init__(self, message):
    self.message = message
    super().__init__(self.message)

#Defining the class to extract the best parameters accoring to the Optuna 
# algorithm used. 
class OptunaParamsExtractor():
  def __init__(self, optuna_optimizer, optuna_tpe=False, optuna_nsga=False):
    self.optuna_tpe = optuna_tpe
    self.optuna_nsga = optuna_nsga
    self.optuna_optimizer = optuna_optimizer
    self.learning_rate = None

    if optuna_tpe==True and optuna_nsga==True:
      raise MultipleTrueError("Only one Optuna algorithm can be set to 'True'")

  def extractor(self, trial_choice=None):
    units = {}
    if self.optuna_tpe:
      if isinstance(trial_choice, int):
        raise TrialChoiceError("'trial_choice' should not be set to any value if Optuna has TPE active.")

      best_params = self.optuna_optimizer.study.best_params
      for param in best_params:
        units.update({param: best_params[param]})

      self.learning_rate = units.pop("learning_rate")
      return units

    if self.optuna_nsga:
      if isinstance(trial_choice, int) == False:
        raise TrialChoiceError("'trial_choice' should be 'int' type when Optuna has NSGAIISampler active.")


      best_trials = self.optuna_optimizer.study.best_trials
      trial_numbers = [num.number for num in best_trials]

      if trial_choice not in trial_numbers:
        raise TrialSelectionError("'trial_choice' should match the actual trial number from the Pareto optimal solutions.")

      for param in best_trials:
        units[f'Trial {param.number}'] = {}
        units[f'Trial {param.number}'].update(param.params)
      
      self.learning_rate = units[f'Trial {trial_choice}'].pop('learning_rate')
      return units[f"Trial {trial_choice}"]