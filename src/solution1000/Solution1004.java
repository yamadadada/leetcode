package solution1000;

public class Solution1004 {

    /**
     * 1004. 最大连续1的个数 III
     * @param A
     * @param K
     * @return
     */
    public int longestOnes(int[] A, int K) {
        int res = 0;
        int left = 0;
        int right = 0;
        int count = 0;
        while (right < A.length) {
            if (A[right] == 0) {
                count++;
            }
            if (count <= K) {
                res = Math.max(res, right - left + 1);
            } else {
                if (A[left] == 0) {
                    count--;
                }
                left++;
            }
            right++;
        }
        return res;
    }

    public static void main(String[] args) {

    }
}
