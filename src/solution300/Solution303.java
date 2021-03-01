package solution300;

public class Solution303 {

    /**
     * 303. 区域和检索 - 数组不可变
     */
    class NumArray {

        private int[] sums;

        public NumArray(int[] nums) {
            sums = new int[nums.length + 1];
            int sum = 0;
            for (int i = 0; i < nums.length; i++) {
                sum += nums[i];
                sums[i + 1] = sum;
            }
        }

        public int sumRange(int i, int j) {
            return sums[j + 1] - sums[i];
        }
    }

}
