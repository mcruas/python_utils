
# Calculates size after merge, without merging
def merge_size(left_frame, right_frame, group_by, how='inner'):
    left_groups = left_frame.groupby(group_by).size()
    right_groups = right_frame.groupby(group_by).size()
    left_keys = set(left_groups.index)
    right_keys = set(right_groups.index)
    intersection = right_keys & left_keys
    left_diff = left_keys - intersection
    right_diff = right_keys - intersection

    left_nan = len(left_frame[left_frame[group_by] != left_frame[group_by]])
    right_nan = len(right_frame[right_frame[group_by] != right_frame[group_by]])
    left_nan = 1 if left_nan == 0 and right_nan != 0 else left_nan
    right_nan = 1 if right_nan == 0 and left_nan != 0 else right_nan

    sizes = [(left_groups[group_name] * right_groups[group_name]) for group_name in intersection]
    sizes += [left_nan * right_nan]

    left_size = [left_groups[group_name] for group_name in left_diff]
    right_size = [right_groups[group_name] for group_name in right_diff]
    if how == 'inner':
        return sum(sizes)
    elif how == 'left':
        return sum(sizes + left_size)
    elif how == 'right':
        return sum(sizes + right_size)
    return sum(sizes + left_size + right_size)