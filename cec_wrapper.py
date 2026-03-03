from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022,
    F52022, F62022, F72022, F82022,
    F92022, F102022, F112022, F122022
)

def get_cec2022_function(fid, dim):

    functions = {
        1: F12022,
        2: F22022,
        3: F32022,
        4: F42022,
        5: F52022,
        6: F62022,
        7: F72022,
        8: F82022,
        9: F92022,
        10: F102022,
        11: F112022,
        12: F122022
    }

    return functions[fid](ndim=dim)