from collections import Counter

def most_frequent_element(arr):
    # Kiểm tra nếu mảng rỗng, ném ra ngoại lệ
    if len(arr) == 0:
        raise ValueError("Mảng không có phần tử nào.")
    
    # Đếm số lần xuất hiện của từng phần tử
    count = Counter(arr)
    
    # Tìm tần suất xuất hiện lớn nhất
    max_frequency = max(count.values())
    
    # Lọc ra các phần tử có tần suất xuất hiện bằng max_frequency
    candidates = [num for num, freq in count.items() if freq == max_frequency]
    
    # Trả về phần tử nhỏ nhất trong các phần tử có tần suất lớn nhất
    return min(candidates)