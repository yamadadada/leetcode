package solution800;

import java.util.HashSet;
import java.util.Set;

public class Solution888 {

    /**
     * 888. 公平的糖果棒交换
     * @param A
     * @param B
     * @return
     */
    public int[] fairCandySwap(int[] A, int[] B) {
        int sumA = 0;
        int sumB = 0;
        Set<Integer> setB = new HashSet<>();
        for (int a : A) {
            sumA += a;
        }
        for (int b : B) {
            sumB += b;
            setB.add(b);
        }
        int c = (sumA - sumB) / 2;
        int[] ans = new int[2];
        for (int a : A) {
            if (setB.contains(a - c)) {
                ans[0] = a;
                ans[1] = a - c;
                return ans;
            }
        }
        return ans;
    }
}
