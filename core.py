__maintainer__ = "Rémi Piché-Taillefer"
__status__ = "1.1.0 reviewed - Added methods to Protocol - Added custom fold compatibility - Restructured 'broad' parameter"
__version__ = "1.1.3a"

""" This library contains functions used to effectively optimize hyperparameters. It was first conceived
for XGBoost, but any machine learning problem, from regression to classification, could make use of
these functions. It is built to be as flexible as possible, yet be able to run with very little
user input. Precision is still proportional to the runtime, but a user can adjust it according to
their needs."""

import numpy as np
import pandas as pd
import pygam as pyg
import xgboost as xgb
from itertools import product
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

#   .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--. 
# :::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\
# '      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `


class Protocol:
    """ Protocol object is used only in `xgboost_smart_search`. It allows a user to skip certain parameter tuning, 
    modify the checked values, its scope, etc. When changing the protocol, refer to `searchCV` for manual range 
    and to `tune_variable` otherwise.

    Usage:
        >>> import hyper_smartsearch as hyp
        >>> p = hyp.Protocol()
        >>> p.depth_child.skip = True
        >>> p.gamma.skip, p.gamma.cv, p.gamma.manual_range = False, 4, np.arange(0, 4, 1)  # Doesn't modify other arguments
        >>> print(p)

    And then pass the object p in the arguments of these functions """
    
    def __init__(self):
        self.n_estimator = self.step(['n_estimators'], "self.n_estimator")
        self.depth_child = self.step(['max_depth', 'min_child_weight'], "self.depth_child")
        self.gamma = self.step(['gamma'], "self.gamma")
        self.subsample_coltree = self.step(['subsample', 'colsample_bytree'], "self.subsample_coltree")
        self.alpha_lambda = self.step(['reg_alpha', 'reg_lambda'], "self.alpha_lambda")
        self.lrate = self.step(['learning_rate'], "self.lrate")

        self.step_list = [self.n_estimator, self.depth_child, self.gamma, 
                          self.subsample_coltree, self.alpha_lambda, self.lrate]    
        self.set_lookup_grid(7, 1.8)
        self.set_psteps(2)

    def skip_all(self, skip=True):
        for stp in self.step_list:
            stp.skip = skip
    
    def set_cv(self, cv):
        for stp in self.step_list:
            stp.cv = cv
            
    def set_lookup_grid(self, len_lookup_grid, last_lookup_grid):
        for stp in self.step_list:
            stp.lookup_grid = [len_lookup_grid, last_lookup_grid]
            stp._manual_range = None
            stp.n_iter = None
            
    def set_psteps(self, v):
        for stp in self.step_list:
            stp.psteps = v
    
    def __str__(self):
        """ Sets signature to reveal the content of the protocol. """
        return "\n".join([s.__str__() for s in self.step_list])

    class step:
        """ This creates default values to every steps of the protocol. """
        def __init__(self, varname, att_name, skip=False, cv=4, manual_range=None, n_iter=None,
                     len_lookup_grid=None, last_lookup_grid=None, psteps=None):
            self.varname = varname
            self.att_name = att_name
            self.skip = skip
            self.len_lookup_grid = len_lookup_grid
            self.last_lookup_grid = last_lookup_grid
            self.psteps = psteps
            self.cv = cv
            self._manual_range = manual_range
            self.n_iter = n_iter
           
        def __str__(self):
            """ Sets signature to reveal the content of the step. """
            signature = self.att_name + "\n"
            signature += (f"  Var(s): {self.varname}") + "\n"
            signature += (f"  skip:   {self.skip}") + "\n"
            if not self.skip:
                signature += (f"  cv:     {self.cv}") + "\n"
                if self.manual_range is not None:
                    signature += (f"  n_iter: {self.n_iter}") + "\n"
                    signature += (f"  manual_range: {self.manual_range}") + "\n"
                elif self.len_lookup_grid is not None:
                    signature += (f"  len_lookup_grid:  {self.len_lookup_grid}") + "\n"
                    signature += (f"  last_lookup_grid: {self.last_lookup_grid}") + "\n"
                    signature += (f"  psteps: {self.psteps}") + "\n"

            return signature
         
            
        @property
        def manual_range(self):
            return self._manual_range
        
        @manual_range.setter
        def manual_range(self, mrange):
            """ Insures that the range becomes a dictionary and gives default values to 'n_iter'. """
            
            dict_mrange = mrange
            if mrange is not None:
                self.last_lookup_grid = None
                self.len_lookup_grid = None
                self.psteps = None
                
                # Tranforms list to dictionary.
                if type(mrange) == list:
                    dict_mrange = {} 
                    # If we are given >=2 lists to put parse in a dictionary
                    if type(mrange[0]) == list:
                        assert len(self.varname) == len(mrange), 'Revise argument in manual range. Should be a dict.'
                        for i, var in enumerate(self.varname):
                            dict_mrange = {**dict_mrange, var: mrange[i]}
                    else:
                        dict_mrange = {self.varname[0]: mrange}
                if self.n_iter is None:
                    self.n_iter = np.product([len(v) for (x, v) in dict_mrange.items()])
            self._manual_range = dict_mrange


        @property
        def lookup_grid(self):
            return self.len_lookup_grid, self.last_lookup_grid

        @lookup_grid.setter
        def lookup_grid(self, v):
            """ Deals with default values. """

            assert type(v) == list and len(v) == 2, "Expect [length of grid, last grid relativity], i.e. [7, 1.8]"
            self.len_lookup_grid, self.last_lookup_grid = v
            if v is not None and not v == [None, None]:
                self._manual_range, self._n_iter = None, None
                if self.psteps is None: self.psteps = 2        


def overnight_tuning(Xt, yt, initial_params, structure, protocol=None, num_rounds=6, weights=None):
    """Loops an XGBoost smartsearch, starting over from the previously optimized hyperparameters.
    The parameters are the same you'd give to the smart search, plus the number of loops.
    See the default protocol and structure in 'xgboost_smart_search' for a standard example.

    Args:
        Xt (pandas DataFrame): training variables
        yt (pandas DataFrame): response variable
        initial_params (dict): parameters to be passed to the estimator, e.g. {'lam':4.0}
        structure (dict): contains 'estimator' (with fit and predict propreties) and 'scoring';
                either a string if recognized by the estimator, or callable(y, y_pred, weights=None)
        protocol (Protocol object): Please... refer to class Protocol to see how to modify it. Leave empty
                otherwise for default xgboost protocol.
        num_rounds (int): number of times the protocol will be looped
        weights (pandas DataFrame): corresponding weights

    Returns:
        (dict) optimized parameters. """

    params = initial_params
    for _ in range(num_rounds):
        params = xgboost_smart_search(Xt, yt, params, structure, protocol, weights)

    # If our scoring is a custom function, use hyper_smartsearch.tune_variable. For native scoring, xgb.cv
    # offers compatible framework, as it accepts strings that refer to known standard metrics.
    print('Final step: Checking for optimal final number of trees.')

    searchCV = Searcher(Xt, yt, structure, weights)
    params['n_estimators'] = searchCV.optimize(_to_dict_of_lists(params), 5, return_trees=True)

    print(f"Best trees: {params['n_estimators']}")
    return params


def xgboost_smart_search(Xt, yt, initial_params, structure=None, protocol=None, weights=None):
    """xgboost_smart_search executes a protocol to optimize sequentially the hyperparameters for xgboost.
    See Protocol class for how to customize the protocol.

    Args:
        Xt (pandas DataFrame): training variables
        yt (pandas DataFrame): response variable
        initial_params (dict): parameters to be passed to the estimator, e.g. {'lam':4.0}
        structure (dict): contains 'estimator' (with fit and predict propreties) and 'scoring';
                either a string if recognized by the estimator, or callable(y, y_pred, weights=None)
        protocol (Protocol object): Please... refer to class Protocol to see how to modify it. Leave empty
                otherwise for default xgboost protocol.
        weights (pandas DataFrame): corresponding weights

    Returns:
        (dict) optimized parameters. """

    p = Protocol() if protocol is None else protocol
    
    default_structure = {'estimator': xgb.XGBRegressor(objective='reg:linear', random_state=123, n_jobs=16, booster='gbtree'),
                         'scoring': mean_squared_error}
    s = default_structure if structure is None else structure

    params = initial_params
    for protocol_step in p.step_list:
        if not protocol_step.skip:
            params = _process_protocol(Xt, yt, s, weights, params, protocol_step)
    return params


def _process_protocol(Xt, yt, structure, weights, params, ps):
    """Called exclusively through 'xgboost_smart_search', executes the steps described in the protocol.
    Returns a dictionary of optimized parameters"""

    print('_____________________________________')
    print(" x ".join(ps.varname))

    if ps.manual_range is None:
        print(f"len_lookup_grid: {ps.len_lookup_grid} - Precision steps: {ps.psteps}")
        variable_tuner = Variable_tuning(Xt, yt, structure, weights)

        return variable_tuner.train(ps.varname, [params[v] for v in ps.varname], params=params, n_steps=ps.psteps, cv=ps.cv, 
                                    len_lookup_grid=ps.len_lookup_grid, last_lookup_relat=ps.last_lookup_relat)

    else:
        if ps.n_iter is None:
            print('Manual range')
        else:
            print(f"Random searching for {ps.n_iter} iterations")
        p_grid = _to_dict_of_lists(params)
        for var in ps.varname:
            p_grid[var] = ps.manual_range[var]

        searchCV = Searcher(Xt, yt, structure, weights)
        params, _ = searchCV.optimize(p_grid, ps.cv, ps.n_iter)
        print([params[v] for v in ps.varname])
        return params


def _to_dict_of_lists(params):
    return dict([[k, [params.get(k)]] for k in params])


#   .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--. 
# :::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\
# '      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `

class Grid:
    """ Grid only check hyperparameter values that are deemed plausible and worth exploring.
    For example, if the best alpha parameter was 0.2 out of [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1]
    then it's probably no use checking for values greater than 1. The grid of explored values is
    constructed and entirely defined by 'num_units' and 'scale_units'. Once a value different from the
    bounds is chosen, we can repeat the same process on a more precise scale, 'n_steps' times."""

    def __init__(self, params=None):
        """ Makes sure 'params' won't be overriden. Since these correspond to the parameters that aren't currently
        tuned, the definition of global variable apply to them.

        Args:
            params (dict): parameters to be passed to the estimator. """

        if params is None:
            params = {}
        self.params = _to_dict_of_lists(params)

    def construct(self, variables, initial_variables, num_units, scale_units, search_direction):
        """ Builds the dictionary of lists.

        Args:
            variables (list of strings): variables to optimize
            initial_variables (list of num): corresponding values
            num_units (int): number of parameter values checked per direction
            scale_units (float): how wide appart the parameter values are to eachother
            search_direction (list of strings): corresponding search direction

        Returns:
            (dict) parameter with associated array of values to evaluate

        """
        local_params = {}

        for var, search_direction_, initial_variable in zip(variables, search_direction, initial_variables):
            cum_scale = self.cumulative_scale(num_units, scale_units, search_direction_)
            updated_unit = initial_variable * cum_scale
            local_params[var] = np.unique([_hyp_adjust(x, var, 0.001) for x in updated_unit]).tolist()


        return {**self.params, **local_params}

    def cumulative_scale(self, num_units, scale_units, search_direction=None):
        """ Solely used by 'create_grid'. Constructs a scale with a wider space between for elements that
        are further from the initial value.

        Args:
            num_units (int): number of parameter values checked per direction
            scale_units (float): how wide apart the parameter values are to eachother

        Returns:
            (numpy array) cumulative scale """

        increasing_scale = np.cumsum([float(x)*scale_units for x in np.arange(1, num_units)]) + 1

        # We want to be able to remove left/right part accordingly, and keep them both if =='both'
        left, right = 1/increasing_scale[::-1], increasing_scale
        if search_direction == 'left':
            right = []
        if search_direction == 'right':
            left = []
        return np.concatenate([left, [1], right])


def _hyp_adjust(x, v, maximum):
    """Makes sure parameter type and range fit expected values.

    Args:
        x (num): value to adjust
        v (string): variable name
        maximum (num): maximum value

    Returns:
        (num) adjusted value """

    if v in ['max_depth', 'n_estimators', 'min_child_weight']:
        return max(1, int(round(x)))
    if v in ['n_splines']:
        return max(4, int(round(x)))
    if v in ['subsample', 'colsample_bytree']:
        return max(0, min(1, x))
    return max(maximum, round(x, 6))


#   .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--. 
# :::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\
# '      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `

class Searcher:
    """    `Searcher` is a fun generalisation of sklearn's GridSearchCV / RandomizedSearchCV. Since GAMs (and
    possibly other models) cannot use these, this function offers a way to do the cross-validation in a grid 
    search, or randomly, naively. It could be used on its own, but a user should be aware of the expected 
    format of the 'structure' and the 'param_grid' input. See 'xgboost_smartSearch' for an standard 
    structure example. """

    def __init__(self, Xt, yt, structure, weights=None, folds=None):
        """ Instantiates a Searcher object.

        Args:
            Xt (pandas DataFrame): training variables;
            yt (pandas DataFrame): response variable;
            structure (dict): contains 'estimator' (with fit and predict propreties) and 'scoring';
                either a string if recognized by the estimator, or callable(y, y_pred, weights=None).
                XGBoost estimators can have an 'early_stop' value for early_stopping. In that case, 
                the scoring field is used as an early stop metric;
            weights (numpy array): corresponding weights;
            folds (numpy array): corresponding fold.

        Returns:
            (None) """

        self.folds = folds
        
        if type(structure['estimator']) in [type(xgb.XGBRegressor()), type(xgb.XGBClassifier())]:
            self.callback = self._searching_xgboost
            self.DM = xgb.DMatrix(Xt, yt, weight=weights, nthread=16)
            self.xgb_params = structure['estimator'].get_xgb_params()
            self._early_stop, self._metric = self._default_early_stop(structure)
            self._metric = self._metric_xgb_decorator(self._metric)

        elif callable(structure['scoring']):
            self.callback = self._searching_naive
            self._uniform_array = np.random.rand(len(yt))
            self._estimator = structure['estimator']
            self._scoring = structure['scoring']
            self._params = self._estimator.get_params()
            self.Xt, self.yt, self.weights = Xt, yt, weights

        else:
            self.callback = self._searching_sklearn
            self._commun_params = {'estimator': structure['estimator'], 'scoring': structure['scoring']}
            self.Xt, self.yt, self.weights = Xt, yt, weights

            
    def optimize(self, param_grid, nfold, n_iter=None, return_trees=False):
        """ According to the originally instantiated object and round-specific parameters, 
        this function applies the optimization procedure.

        Args:
            param_grid (dict): dictionary of variables to be passed to the estimator, where every
                    element is a numpy array of values to be evaluated;
            nfold (int): number of cross-validation folds. If 'folds' from __init__ is not None, this HAS TO BE
                    coherent with how many different group there are in 'folds';
            n_iter (Maybe int): if None, will go through every combination of parameters.
                    Otherwise, will only choose n_iter of them, randomly.

        Returns:
            (dict) best parameter combination found amongst param_grid; 
            (float) best corresponding score. """

        self.param_grid = param_grid
        self.flat = _flatten_dict(param_grid)
        grid_size = len(self.flat)

        if n_iter is not None:
            self.order_of_operation = np.argsort(np.random.rand(grid_size))
            n_iter = min(n_iter, grid_size)
        else:
            self.order_of_operation = np.array(range(grid_size))
            n_iter = grid_size

        res = self.callback(n_iter, nfold)
        return self.conclude(res, return_trees)

    
    def conclude(self, res, return_trees):
        """ Prints the results of this round and parses through branch-specific outputs. """

        output_print = "  error: [%.6f]" % (res['_min_score'])
        if '_min_trees' in res:
            output_print += f", ntrees: {res['_min_trees']}"
        print(output_print)

        if return_trees:
            return res['_min_trees']
        return res['_min_params'], res['_min_score']
    
    
    def _fold_params(self, nfold):
        """ Insures that the fold format is exactly what XGBoost/SKLearn needs."""
        if self.folds is None:
            return {'nfold': nfold}
        else:
            gkf = GroupKFold(nfold)
            return {'folds': [x for x in gkf.split(self.folds, groups=self.folds)]}

        
    def _searching_xgboost(self, n_iter, nfold):
        """ Cross-validation in xgboost is best done through its function `xgb.cv`. Notice that the
        objective function will be passed through the parameters, which then differentiates between
        regression / classification problem. Early stopping is optional, as `_early_stop` will be 
        assigned a default None value if none is given. The number of trees used will be printed
        afterward, but isn't used elsewhere for any other purpose. """

        grid_score = []
        grid_trees = []
        for index in range(n_iter):
            overriding_params = self.flat[self.order_of_operation[index]]
            xgb_params = _update_params(self.xgb_params, overriding_params)
            fold_param = self._fold_params(nfold)

            xgcv = xgb.cv(xgb_params, self.DM, num_boost_round=xgb_params['n_estimators'], feval=self._metric,
                          maximize=False, show_stdv=False, early_stopping_rounds=self._early_stop, **fold_param)

            # The score for these particular hyperparameters is stored on the last line, under 'test-<metric>-mean' (1st col)
            grid_score += [xgcv.iloc[-1][0]]
            grid_trees += [xgcv.shape[0]]
            
        return {
            '_min_score': np.min(grid_score),
            '_min_params': self.flat[self.order_of_operation[np.argmin(grid_score)]],
            '_min_trees': grid_trees[np.argmin(grid_score)]
        }
    

    def _searching_naive(self, n_iter, nfold):
        """ For other estimators like GAMs, naive grid search is prefered, as Sklearn's GridSearch doesn't
        encapsulates them correctly. Naive grid search (or random search) uses cross validation and returns
        the parameters that performed the best on average on the out-of-fold. """
        
        grid_score = []
        for index in range(n_iter):

            params = self.flat[self.order_of_operation[index]]
            estimator = self._estimator.set_params(**_update_params(self._params, params))

            if self.folds is None:
                self.folds = [int(x*nfold) for x in self._uniform_array]
            
            scorecv = []

            for i in range(nfold):
                # We want to fit on the training set, predict on the validation set, and evaluate validation's score
                valid_index = np.array([x == i for x in self.folds])

                # If we don't have weights, sending a None will still cause an error if we're not expecting it.
                if self.weights is None:
                    kwargs, kwargs_fit = {}, {}
                else:
                    # Trying to find how the estimator calls its weights.
                    w_v_scoring = _get_weightname(self._scoring.__code__.co_varnames)
                    w_e_scoring = _get_weightname(estimator.fit.__code__.co_varnames)
                    kwargs = {w_v_scoring: self.weights.loc[valid_index]}
                    kwargs_fit = {w_e_scoring: self.weights.loc[~valid_index]}

                estimator.fit(self.Xt.loc[~valid_index], self.yt[~valid_index], **kwargs_fit)

                pred = estimator.predict(self.Xt.loc[valid_index])
                scorecv += [self._scoring(self.yt[valid_index], pred, **kwargs)]

            grid_score += [np.mean(scorecv)]

        return {
            '_min_score': np.min(grid_score),
            '_min_params': self.flat[self.order_of_operation[np.argmin(grid_score)]]
        }
    

    def _searching_sklearn(self, n_iter, nfold):
        """ For all other estimators, SKlearn is a solid option. This function encapsulates the grid and the
        random search in a very simple fashion. """

        self._commun_params['scoring'] = self._metric_sklearn_decorator(self._commun_params['scoring'])
        fold = self._fold_params(nfold)
        
        if n_iter is None:
            sk_est = GridSearchCV(param_grid=self.param_grid, cv=fold, **self._commun_params)
        else:
            sk_est = RandomizedSearchCV(param_distributions=self.param_grid, n_iter=n_iter, cv=fold, **self._commun_params)
        sk_est.fit(self.Xt, self.yt, sample_weight=self.weights)

        return {
            '_min_score': sk_est.best_score_,
            '_min_params': sk_est.best_params_
        }

    
    def _metric_xgb_decorator(self, func_metric):
        """ Wraps a function to the desired format of `xgb.cv`. """
        def new_metric(ypred, dm):
            w = dm.get_weight()
            if len(w) == 0:
                kwargs = {}
            else:
                metrics_args = func_metric.__code__.co_varnames
                kwargs = {_get_weightname(metrics_args): w}
            return 'error', func_metric(dm.get_label(), ypred, **kwargs)
        return new_metric

    
    def _metric_sklearn_decorator(self, metric):
        """ Wraps a function to the desired format of GridSearchCV. Note that sample_weights are not
        yet implemented into SKlearn's scorer. I'd suggest using a metric as a string maybe? """
        if callable(metric):
            return make_scorer(metric)
        return metric

    
    def _default_early_stop(self, structure):
        """ Only used for XGBoost estimators, puts default values to structure."""
        if 'early_stop' not in structure:
            return None, mean_squared_error
        elif 'scoring' not in structure:
            return structure['early_stop'], mean_squared_error
        else:
            return structure['early_stop'], structure['scoring']


def _update_params(original_params, overriding_params):
    """ Updates parameters while not overriding directly on the original object. """
    results = original_params
    for x in overriding_params:
        results[x] = overriding_params[x]
    return results


def _flatten_dict(param_grid):
    """ Transforms a dictionary into a list of all possible combinations. """
    flat = [[(k, v) for v in vs] for k, vs in param_grid.items()]
    return [dict(items) for items in product(*flat)]


def _get_weightname(variables):
    """Trying to find how a function calls its weight argument"""
    for xname in ['weights', 'weight', 'sample_weight', 'sample_weights', 'w', 'wt']:
        if xname in variables:
            return xname

    print("Could not proprely put weights into estimator. Estimator.fit did not have an argument for \
          'weights', 'weight', 'sample_weight', 'sample_weights' nor 'w'.")
    return variables[2]
            

#   .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--. 
# :::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\
# '      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `

class Variable_tuning:
    """ Variable_tuning isn't tied to xgboost and can/should be used for any tuning needed. It'll use the
    inital parameters and gradually find ones that perform the best through cross-validation. It can
    take as an input any number of hyper-parameters to tune together (input variable 'var'), as long as
    it received a vector of initial values of the same size (input variable 'initial')."""

    def __init__(self, Xt, yt, structure, weights=None, folds=None):
        """ Instantiates the Searcher object.

        Args:
            Xt (pandas DataFrame): training variables
            yt (pandas DataFrame): response variable
            structure (dict): contains 'estimator' (with fit and predict propreties) and 'scoring';
                either a string if recognized by the estimator, or callable(y, y_pred, weights=None)
            weights (numpy array): corresponding weights;
            folds (numpy array): corresponding folds."""

        self.search = Searcher(Xt, yt, structure, weights, folds)

        
    def train(self, var, initial, params=None, n_steps=2, cv=4, len_lookup_grid=None, last_lookup_grid=None, broad=None):
        """ Optimizes defined variables on an estimator.

        Args:
            var (list of strings): variables to optimize simultaneously
            initial (list of num): corresponding initial values
            params (dict): every other parameters to pass to the estimator
            n_steps (int): number of precision steps, 3 being quite precise, 1 being rather fast
            cv (int): number of cross-validation fold
            len_lookup_grid (odd int): length of grid for the search
            last_lookup_grid (float): for an initial value of x, the algorithm will check up from
                x/last_lookup_grid to x*last_lookup_grid, in len_lookup_grid steps.
            broad (int): indicator of the scale with possible values:
                0: Short range
                1: Large range, precise
                2: Wide range, not-precise
                3: Very wide range, precise
                MOSTLY DEPRECIATED. Use "len_lookup_grid" and "last_lookup_grid" instead.

        Local variables:
            search_direction (numpy array): allows the algorithm to only check one direction (bigger or 
                smaller), or both, by default.
            var_todo (list of strings): variables that have not yet been optimized
            var_todo_opt (list of int): corresponding optimized values

        Returns:
            (dict) optimized parameters. """

        self.kN, self.kScale = self.set_grid_dimensions(broad, len_lookup_grid, last_lookup_grid)
        self.last_lookup_grid = last_lookup_grid
        self.previous_params, self.previous_score = None, None
        if params is None:
            params = {}
        params = _update_params(params, dict(zip(var, initial)))

        for step in range(n_steps):
            print(f"Step {step+1}: ")

            search_direction = np.repeat('both', len(var))
            var_todo = np.array(var)
            var_todo_opt = np.array([params.get(x) for x in var])

            while True:
                gd = Grid(params).construct(var_todo, var_todo_opt, self.kN, self.kScale, search_direction)
                number_of_disp = np.prod([len(v) for (x, v) in gd.items()])
                if number_of_disp == 1:
                    break

                print(f" <cross-validating for {number_of_disp} dispositions>")
                var_todo_initial_values = var_todo_opt
                var_todo_opt = self._train_iteration(gd, var_todo, cv)

                # ceiled_variable : list of variables that were out of scope
                ceiled_variable, search_direction, params = self._check_bounds(
                    params, var_todo, var_todo_initial_values, var_todo_opt)
                if len(ceiled_variable) == 0:
                    break

                # Updating the list of variables that still need tuning
                var_todo = var_todo[ceiled_variable]
                var_todo_opt = var_todo_opt[ceiled_variable]

            # We reduce the scale after each step, gradually converging
            self.kScale /= self.kN
        return params


    def set_grid_dimensions(self, broad=None, len_lookup_grid=None, last_lookup_relat=None):
        """ Unifies all different definition of the dimensions. These refer to the values checked at each
        iterations, e.g. the values that define [0.7692, 0.9091, 1.0, 1.1, 1.3] are its symmetrical length
        (len_grid: 5) and its last relativity (last_relat: 1.3). Computes from them the increment and the
        number of half-length. """

        if broad is not None:
            assert broad in range(4), "Broad value invalid: Should be between 0 and 3."
            _kN = [3, 5, 4, 7][broad]
            _kScale = [0.1, 0.1, 0.15, 0.1][broad]
        else:
            _kN = np.ceil((len_lookup_grid - 1)/2) + 1
            _kScale = 2 * (last_lookup_relat - 1)/(_kN * (_kN - 1))
        return _kN, _kScale
            
        
    def _train_iteration(self, gd, var_todo, cv):
        """ Applies the optimisation, and makes sure we at least improved our performances. """

        gdcv, score = self.search.optimize(gd, cv)
        previous_not_set = self.previous_params is None or self.previous_score is None
        if previous_not_set or score < self.previous_score:
            self.previous_params, self.previous_score = gdcv, score
        else:
            gdcv = self.previous_params

        print(" ", ", ".join([f"{x}: {gdcv[x]}" for x in var_todo]))
        return np.array([gdcv[x] for x in var_todo])

    
    def _check_bounds(self, params, var_todo, var_todo_initial_values, var_todo_opt):
        """Deals with bounds, and determine whether the scope must be extended"""

        ceiled_variable, new_search_direction = list(), list()
        inner_params = params
        for index, var in enumerate(var_todo):
            ivar_ini, ivar_opt = var_todo_initial_values[index], var_todo_opt[index]
            adj_low, adj_high, adj_opt, adj_ini = self._get_bounds(var, ivar_opt, ivar_ini)
            if adj_low == adj_high:
                inner_params[var] = adj_low
                continue

            # If we reached the bottom of the set, we wanna look at only smaller values
            # If we reached the top of the set, we wanna look at only larger values
            if adj_opt == adj_low and adj_low != adj_ini:
                ceiled_variable.append(index)
                new_search_direction.append('left')
            elif adj_opt == adj_high and adj_high != adj_ini:
                ceiled_variable.append(index)
                new_search_direction.append('right')
            
            # If the range was sufficient, we do not wish to retry again every possible values.
            # It will still be fine-tuned later, in the next 'step'.
            inner_params[var] = adj_opt
        return ceiled_variable, new_search_direction, inner_params

    
    def _get_bounds(self, var, var_opt, var_ini):
        """Gets bounds from the cumulative scale and adjusts them accordingly"""

        lower_bound = var_ini / self.last_lookup_grid
        higher_bound = var_ini * self.last_lookup_grid
        v = [lower_bound, higher_bound, var_opt, var_ini]
        return list(map(lambda x: _hyp_adjust(x, var, 0), v))

#   .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--.      .--. 
# :::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\
# '      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `--'      `


