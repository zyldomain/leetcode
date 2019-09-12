
public class Main2 {
	public static void duipai(int[] array){
        //首先建立初始堆
        for(int i = 1 ; i <= array.length ; i++){
            int index = i;
            while(index / 2 > 0){
                if(array[index - 1] > array[index / 2 - 1]){
                    int tmp = array[index - 1];
                    array[index - 1] = array[index / 2 - 1];
                    array[index / 2 - 1] = tmp;
                    index /= 2;
                }else{
                    break;
                }
            }
        }
        System.out.println("...");
        int index = array.length - 1;
        while(index > 0){
            int tmp = array[0];
            array[0] = array[index];
            array[index--] = tmp;
            //修正
            int i = 1;
            while(i <= index + 1){
                if(2 * i <= index + 1 && 2 * i + 1 <= index + 1){
                    if(array[2 * i - 1] >= array[2 * i] && array[2 * i - 1] > array[i - 1]){
                        tmp = array[2 * i - 1];
                        array[2 * i - 1] = array[i - 1];
                        array[i - 1] = tmp;
                        i = 2 * i;
                    }else if(array[2 * i - 1] < array[2 * i] && array[2 * i] > array[i - 1]){
                        tmp = array[2 * i];
                        array[2 * i] = array[i - 1];
                        array[i - 1] = tmp;
                        i = 2 * i + 1;
                    }
                }else if(2 * i <= index + 1&& array[2 * i - 1] > array[i - 1]){
                    tmp = array[2 * i - 1];
                    array[2 * i - 1] = array[i - 1];
                    array[i - 1] = tmp;
                    i = 2 * i;
                }else{
                    break;
                }
            }
        }
    }
	
	public static int findKthLargest(int[] nums, int k) {
        int[] array = new int[k + 1];
        //建堆
        for(int i = 0 ; i < k ; i++){
            array[i + 1] = nums[i];
            //修改
            int index = (i + 1) / 2;
            while(index > 0){
                if(2 * index + 1 <= i + 1){
                    if(array[index * 2] <= array[index * 2 + 1] && array[index * 2] < array[index]){
                        int tmp = array[index * 2];
                        array[index * 2] = array[index];
                        array[index] = tmp;
                    }else if(array[index * 2] > array[index * 2 + 1] && array[index * 2 + 1] < array[index]){
                        int tmp = array[index * 2 + 1];
                        array[index * 2 + 1] = array[index];
                        array[index] = tmp;
                    }
                }else{
                    if(array[index] > array[index * 2]){
                        int tmp = array[index * 2];
                        array[index * 2] = array[index];
                        array[index] = tmp;
                    }
                }
                index /= 2;
            }
        }

        for(int i = k ; i < nums.length ; i++){
            if(nums[i] > array[1]){
                array[1] = nums[i];
                //修改
                int index = 1;
                while(index <= k){
                    if(2 * index + 1 <= k){
                        if(array[index * 2] <= array[index * 2 + 1] && array[index] > array[index * 2]){
                            int tmp = array[index * 2];
                            array[index * 2] = array[index];
                            array[index] = tmp;
                            index = index * 2;
                        }else if(array[index * 2] > array[index * 2 + 1] && array[index] > array[index * 2 + 1]){
                            int tmp = array[index * 2 + 1];
                            array[index * 2 + 1] = array[index];
                            array[index] = tmp;
                            index = index * 2 + 1;
                        }else{
                            break;
                        }
                    }else if(2 * index <= k){
                        if(array[index * 2] < array[index]){
                            int tmp = array[index * 2 ];
                            array[index * 2 ] = array[index];
                            array[index] = tmp;
                            index = index * 2;
                        }else{
                            break;
                        }
                    }else{
                        break;
                    }
                }
            }
        }
        return array[1];
    }
	public static void main(String[] args) throws Exception {
		System.out.println("hhh");
	}
}
