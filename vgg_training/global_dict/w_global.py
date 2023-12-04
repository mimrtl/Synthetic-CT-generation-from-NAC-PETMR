#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

global gbl_dict
gbl_dict = {}


def gbl_set_value(key, value) -> object:
    """

    :rtype: 
    """
    gbl_dict[key] = value


def gbl_get_value(key, defvalue=None):
    try:
        return gbl_dict[key]
    except KeyError:
        return defvalue


def gbl_save_value():
    model_id = gbl_get_value("model_id")
    np.save(gbl_dict, 'dict_' + model_id + '.npy')


def gbl_all():
    return gbl_dict
