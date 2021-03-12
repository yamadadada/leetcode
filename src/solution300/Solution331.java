package solution300;

public class Solution331 {

    /**
     * 331. 验证二叉树的前序序列化
     * @param preorder
     * @return
     */
    public boolean isValidSerialization(String preorder) {
        String[] array = preorder.split(",");
        int slot = 1;
        for (String s : array) {
            if (slot == 0) {
                return false;
            }
            if ("#".equals(s)) {
                slot--;
            } else {
                slot++;
            }
        }
        return slot == 0;
    }
}
