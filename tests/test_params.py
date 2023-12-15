import igm
import pytest

# from igm.common import validate_module, load_modules_from_directory
from json import JSONDecodeError


def test_load_json_params_with_comments():
    """Tests that adding comments to the json will fail with a decode error."""

    with pytest.raises(JSONDecodeError):
        _ = igm.get_modules_list("params_comments.json")


def test_load_igm_modules():
    """Tests that the core igm modules LIST loaded from the params.json file are loaded."""
    modules_dict = igm.get_modules_list("params.json")

    preproc_modules = modules_dict["modules_preproc"]
    process_modules = modules_dict["modules_process"]
    postproc_modules = modules_dict["modules_postproc"]

    igm_core_preproc_modules = ["load_ncdf"]
    igm_core_process_modules = [
        "iceflow",
        "time",
        "thk",
        "rockflow",
        "vert_flow",
        "particles",
    ]
    igm_core_postproc_modules = ["write_ncdf", "plot2d", "print_info", "print_comp"]

    assert set(igm_core_preproc_modules).issubset(preproc_modules)
    assert set(igm_core_process_modules).issubset(process_modules)
    assert set(igm_core_postproc_modules).issubset(postproc_modules)


def test_load_modules():
    """Tests that the core igm modules LIST AND the custom modules from the params.json file are loaded."""
    modules_dict = igm.get_modules_list(
        "params.json"
    )
    assert modules_dict == {
        "modules_preproc": ["load_ncdf", "track_usurf_obs"],
        "modules_process": [
            "clim_aletsch",
            "smb_accmelt",
            "iceflow",
            "time",
            "thk",
            "rockflow",
            "vert_flow",
            "particles",
        ],
        "modules_postproc": ["write_ncdf", "plot2d", "print_info", "print_comp"],
    }
