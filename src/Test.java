import common.ListNode;
import common.TreeNode;
import utils.ArrayUtil;
import utils.ListNodeUtil;
import utils.ListUtil;

import java.util.List;

public class Test {

    public static void main(String[] args) {
        Solution solution = new Solution();
        ListNode head = ListNodeUtil.generateListNode();
        System.out.println(solution.isPalindrome(head));
    }
}
