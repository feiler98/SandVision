# imports
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def mp_list_to_chunks(input_list: list, n_chunks: int) -> list:
    """
    Multiprocessing list to list-chunks.

    Parameters
    ----------
    input_list: list
    n_chunks: int

    Returns
    -------
    list
        A list of lists as generator for memory-saving.
    """
    size_requirement = int(len(input_list) / 1000)
    step_size = size_requirement if size_requirement < n_chunks else n_chunks
    for i in range(0, len(input_list), step_size):
        yield input_list[i:i+step_size]
