package solution900;

import java.util.HashMap;
import java.util.Map;

public class Solution992 {

    /**
     * 992. K 个不同整数的子数组
     * @param A
     * @param K
     * @return
     */
    public int subarraysWithKDistinct(int[] A, int K) {
        return maxWithKDistinct(A, K) - maxWithKDistinct(A, K - 1);
    }

    private int maxWithKDistinct(int[] A, int k) {
        int left = 0;
        int right = 0;
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        while (right < A.length) {
            if (map.containsKey(A[right])) {
                map.put(A[right], map.get(A[right]) + 1);
            } else {
                map.put(A[right], 1);
            }

            while (map.keySet().size() > k) {
                if (map.get(A[left]) > 1) {
                    map.put(A[left], map.get(A[left]) - 1);
                } else {
                    map.remove(A[left]);
                }
                left++;
            }
            res += right - left + 1;
            right++;
        }
        return res;
    }
}
