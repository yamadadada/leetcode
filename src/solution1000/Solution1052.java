package solution1000;

public class Solution1052 {

    /**
     * 1052. 爱生气的书店老板
     * @param customers
     * @param grumpy
     * @param X
     * @return
     */
    public int maxSatisfied(int[] customers, int[] grumpy, int X) {
        int sum = 0;
        for (int i = 0; i < X; i++) {
            if (grumpy[i] == 1) {
                sum += customers[i];
            }
        }
        int left = 0;
        int right = X;
        int maxSum = sum;
        int maxIndex = 0;
        while (right < customers.length) {
            if (grumpy[right] == 1) {
                sum += customers[right];
            }
            if (grumpy[left] == 1) {
                sum -= customers[left];
            }
            left++;
            if (sum > maxSum) {
                maxSum = sum;
                maxIndex = left;
            }
            right++;
        }
        int res = 0;
        for (int i = 0; i < customers.length; i++) {
            if (i >= maxIndex && i < maxIndex + X) {
                res += customers[i];
            } else {
                if (grumpy[i] == 0) {
                    res += customers[i];
                }
            }
        }
        return res;
    }
}
