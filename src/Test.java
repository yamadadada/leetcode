import common.ListNode;
import common.TreeNode;
import utils.ArrayUtil;
import utils.ListNodeUtil;
import utils.ListUtil;

import java.util.ArrayList;
import java.util.List;

public class Test {

    public static void main(String[] args) {
        int[] array = new int[] {1,2,3,4,5,6,7,8,9};
        Solution solution = new Solution();
        System.out.println(solution.wiggleMaxLength(array));
    }
}
