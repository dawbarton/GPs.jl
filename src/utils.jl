#--- Generic functions used by multiple types

"""
Set the hyperparameters of an object.
"""
function sethyperparameters!
end

"""
Return the hyperparameters of an object.
"""
function gethyperparameters
end

"""
Return the number of hyper parameters an object has.
"""
function hyperparametercount
end

export sethyperparameters!, gethyperparameters, hyperparametercount
