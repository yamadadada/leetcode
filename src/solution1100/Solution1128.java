package solution1100;

public class Solution1128 {

    /**
     * 1128. 等价多米诺骨牌对的数量
     * @param dominoes
     * @return
     */
    public int numEquivDominoPairs(int[][] dominoes) {
        int[] nums = new int[100];
        int res = 0;
        for (int[] dominoe : dominoes) {
            if (dominoe[0] > dominoe[1]) {
                int temp = dominoe[0];
                dominoe[0] = dominoe[1];
                dominoe[1] = temp;
            }
            res += nums[dominoe[0] * 10 + dominoe[1]];
            nums[dominoe[0] * 10 + dominoe[1]]++;
        }
        return res;
    }
}
