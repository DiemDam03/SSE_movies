list1 = [1, 2, 3, 4, 2]
list2 = [3, 5, 6, 1]

# Concatenate the lists
merged_list = list1 + list2
print("merge:",merged_list)

# Convert to a set to remove duplicates
unique_elements_set = set(merged_list)
print("set:",unique_elements_set)

# Convert back to a list
final_list = list(unique_elements_set)

print(final_list)