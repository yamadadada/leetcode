import utils.ArrayUtil;

import java.util.List;

public class Test {

    public static void main(String[] args) {
        Solution solution = new Solution();
        int[] nums = new int[]{4, 6, 7, 7};
        List<List<Integer>> result = solution.findSubsequences(nums);
        System.out.println(result);
    }
}
