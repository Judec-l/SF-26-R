def setParameter(parameter, field, default):
    """
    Mimics MATLAB setParameter for struct fields.
    Parameters
    ----------
    parameter : dict
        Dictionary containing parameters
    field : str
        Field name
    default : any
        Default value if field does not exist

    Returns
    -------
    v : any
        parameter[field] if exists, else default
    """
    return parameter[field] if field in parameter else default

