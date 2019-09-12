import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.locks.StampedLock;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.management.StringValueExp;

import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.SortedMap;
import java.util.Stack;
import java.util.TreeMap;
import java.util.WeakHashMap;

class Solution {
    public static boolean $isMatch(String s, String p) {
        boolean flag = true;
        int i = 0 , j = 0;
        char ch = ' ';
        while(i < s.length() && j < p.length()) {
        	//如果当前的字符一样
        	if(s.charAt(i) == p.charAt(j)) {
        		ch = s.charAt(i);
        		i++;
        		j++;
        	}else {
        		if(j - 1 >= 0) {
        			ch = p.charAt(j - 1);
        		}
        		//如果当前字符不一样,需要判断当前是否是通配符,或者下一个是否是通配符
        		if(p.charAt(j) == '*') {
        			if(ch == '.') {
        				while(i < s.length() && (j + 1 < p.length() && p.charAt(j + 1) != s.charAt(i) || j + 1 == p.length())) {
        					i++;
        				}
        			}else {
        				while(i < s.length() && s.charAt(i) == ch) {
            				i++;
            			}
        			}
        			j++;
        			while(j < p.length() && ch == p.charAt(j))
        				j++;
        		}else if(p.charAt(j) == '.') {
        			ch = '.';
        			i++;
        			j++;
        			
        		}else if(j + 1 < p.length()  && p.charAt(j + 1) == '*') {        			
        			p = p.substring(0, j) + p.substring(j + 2);
        		}else {
        			return false;
        		}
        	}
        }
        if(i < s.length()) {
        	return false;
        }
        boolean skip = false;
        if(j < p.length()) {
        	if(p.charAt(j) == '*') {
        		p = ch + p.substring(j);
        		skip = true;
        	}else if(j - 1 >= 0 && p.charAt(j - 1) == '*') {
        		p = ch + "*" + p.substring(j);
        		skip = true;
        	}else {
        		p = p.substring(j);
        	}
        	
        	j = 0;
        	while(j < p.length()) {
        		if(p.charAt(j) != '*' && j + 1 < p.length() && p.charAt(j + 1) == '*') {
        			p = p.substring(0, j) + p.substring(j + 2);
        		}else {
        			if(skip && j == p.length() - 1 && p.charAt(j) == ch) {
        				return true;
        			}
        			return false;
        		}
        	}
        }
      
        return flag;
    }
    public static int maxArea(int[] height) {
        int max = 0;
        int i = 0 , j = height.length - 1;
        while(i < j) {
        	if((j - i) * (Math.min(height[i], height[j])) > max) {
        		max = (j - i) * (Math.min(height[i], height[j]));
        	}
        	
        	if(height[i] > height[j]) {
        		j--;
        	}else {
        		i++;
        	}
        }
        return max;
    }
    
    public static String longestCommonPrefix(String[] strs) {
    	if(strs == null || strs.length <= 0) {
    		return "";
    	}
        String str = strs[0];
        for(int i = 1 ; i < strs.length ; i++) {
        	//每次和str比较
        	int j = 0;
        	while(j < (Math.min(str.length(), strs[i].length())) && str.charAt(j) == strs[i].charAt(j)) {
        		j++;
        	}
        	if(j < str.length()) {
        		str = str.substring(0, j);
        	}
        	if(j == 0) {
        		return "";
        	}
        }
        return str;
    }
    
    public static List<List<Integer>> threeSum(int[] nums) {
    	List<List<Integer>> results = new ArrayList<>();
    	//数组去重
    	int[] used = new int[nums.length];
    	Arrays.sort(nums);
    	int index = 0;
    	for(int i = 0 ; i < nums.length ; i++) {
    		if(nums[i] >= 0) {
    			if(index == 0 || nums[i] == 0) {
    				index = i;
    				
    			}
    			if(nums[i] > 0) {
					break;
				}
    		}
    	}
    	findSum(index, nums, 0 , used , 0 , -1 , new ArrayList<>(), results);
    	return results;
    }
    
    private static void findSum(int split , int[] nums , int index, int[] used , int flag , int sum , List<Integer> temp , List<List<Integer>> results ) {
    	if(sum == 0 && flag != 0 && temp.size() == 3) {
    		boolean skip = false;
    		int record = 0;
    		for(List<Integer> list : results) {
    			record = 0;
    			for(int i = 0 ; i < 3 ; i++) {
    				if(list.get(i) == temp.get(i)) {
    					record++;
    				}
    			}
    			if(record == 3) {
    				skip = true;
    				break;
    			}
    		}
    		if(!skip) {
    			results.add(new ArrayList<>(temp));
    		}
    		return;
    	}
    	if(index == nums.length || temp.size() >= 3) {
    		return;
    	}
    	List<Integer> list = new ArrayList<>(temp);
    	for(int i = temp.size() == 2 ? index > split ? index : split : index ; i < (temp.size() == 0 ? split : nums.length) ; i++) {
    		if(used[i] == 0) {
    			used[i] = 1;
    			if(flag == 0) {
        			sum = 0;
        		}
    			list.add(flag , nums[i]);
    			findSum(split, nums, i + 1, used, flag + 1, sum + nums[i], list, results);
    			list.remove(flag);
    			used[i] = 0;
    		}
    		
    	}
    }
    static int max = 0;
    static int value = 0;
    public static int threeSumClosest(int[] nums, int target) {
        int[] used = new int[nums.length];
        value = nums[0] + nums[1] + nums[2];
        max = Math.abs(value - target);
        findMax(0, used, nums, 0, 0, target);
        return value;
    }
    private  static void findMax(int index , int[] used , int[] nums, int count, int sum , int target) {
    	if(count == 3) {
    		if(Math.abs(sum - target) < max) {
    			max = Math.abs(sum - target);
    			value = sum;
    		}
    		return;
    	}
    	for(int i = index ; i < nums.length ; i++) {
    		if(used[i] == 0) {
    			used[i] = 1;
    			findMax(i + 1, used, nums, count + 1, sum + nums[i], target);
    			used[i] = 0;
    		}
    	}
    	
    }
    
    public static List<String> letterCombinations(String digits) {
    	Map<String, String> map = new HashMap<>();
    	map.put("2", "abc");
    	map.put("3", "def");
    	map.put("4", "ghi");
    	map.put("5", "jkl");
    	map.put("6", "mno");
    	map.put("7", "pqrs");
    	map.put("8", "tuv");
    	map.put("9", "wxyz");
    	String[] strs = digits.split("");
    	List<String> list = new ArrayList<>();
    	findAllString(map, strs, 0, "", strs.length, list);
    	return list;        
    }
    private static void findAllString(Map<String, String> map , String[] digits , int index , String recordString , int length , List<String> list) {
    	if(index == length) {
    		list.add(recordString);
    		return ;
    	}
    	char[] content = map.get(digits[index]).toCharArray();
    	
    	for(int i = 0 ; i < content.length ; i++) {
    		findAllString(map, digits, index + 1, recordString + content[i], length, list);
    	}
    	
    }
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if(head == null || (head.next == null && n == 1)) {
        	return null;
        }
        ListNode p = head;
        int i = 0;
        for(; i < n - 1 ; i++) {
        	p = p.next;
        	if(p == null) {
        		return null;
        	}       	
        }
        
        ListNode q = head;
        ListNode pre = null;
        while(p.next != null) {
        	pre = q;
        	q = q.next;
        	p = p.next;
        }
        if(pre == null) {
        	return q.next;
        }else {
        	pre.next = q.next;
        }
        return head;
    }
    
    public static  boolean isValid(String s) {
        boolean flag = true;
        if(s != null && s.length() > 0) {
        	if(s.length() % 2 != 0) {
        		return false;
        	}else {
        		int i = 0 ;
        		int j = -1;
        		char[] array = new char[s.length()];
        		while(i < s.length()) {
        			if(j == -1 ||( array[j] != s.charAt(i) - 1 && array[j] != s.charAt(i) - 2)) {
        				array[++j] = s.charAt(i);
        			}else {
        				j--;
        			}
        			i++;
        		}
        		if(j != -1) {
        			flag = false;
        		}
        	}
        }
        return flag;
    }
    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null && l2 == null) {
        	return null;
        }
        if(l1 == null) {
        	return l2;
        }
        if(l2 == null) {
        	return l1;
        }
        ListNode head = null , p = l1 , q = l2 , tail = null;
        while(p != null && q != null) {
        	if(p.val < q.val) {
        		if(head == null) {
        			head = p;
        			tail = head;
        		}else {
        			tail.next = p;
        			tail = p;
        		}
        		p = p.next;
        	}else {
        		if(head == null) {
        			head = q;
        			tail = head;
        		}else {
        			tail.next = q;
        			tail = q;
        		}
        		q = q.next;
        	}
        	
        	
        }
        while(p != null) {
        	tail.next = p;
        	tail = tail.next;
        	p = p.next;
        }
        
        while(q != null) {
        	tail.next = q;
        	tail = tail.next;
        	q = q.next;
        }
        
        tail.next = null;
        return head;
        
    }
    
    public static List<String> generateParenthesis(int n) {
        List<String> results = new ArrayList<>();
        char[] source = new char[2 * n];
        for(int i = 0 ; i < 2 * n ; i += 2) {
        	source[i] = '(';
        	source[i + 1] = ')';
        }
        Set<String> sets = new HashSet<>();
        results.add(new String(source));
        produceKH(results, sets, source);
        return results;
    }
    private static void produceKH(List<String> results, Set<String> sets, char[] source) {
    	for(int i = 0 ; i < source.length ; i++) {
    		if(source[i] == ')') {
    			for(int j = i + 1 ; j < source.length ; j++) {
        			if(source[j] == '(') {
        				
        				source[j] = ')';
        				source[i] = '(';
        				String target  = new String(source);
        				if(!sets.contains(target)) {
        					sets.add(target);
        					results.add(target);
        					produceKH(results, sets, source);
        				};
            				
        				source[i] = ')';
        				source[j] = '(';
        			}
        		}
    		}
    		
    	}
    	
    }
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists == null || lists.length == 0) {
        	return null;
        }
        if(lists.length == 1) {
        	return lists[0];
        }
        
        ListNode head = lists[0];
        for(int i = 1 ; i < lists.length ; i++) {
        	head = merge(head, lists[i]);
        }
        return head;
    }
    
    public static ListNode merge(ListNode l1, ListNode l2) {
        if(l1 == null && l2 == null) {
        	return null;
        }
        if(l1 == null) {
        	return l2;
        }
        if(l2 == null) {
        	return l1;
        }
        ListNode head = null , p = l1 , q = l2 , tail = null;
        while(p != null && q != null) {
        	if(p.val < q.val) {
        		if(head == null) {
        			head = p;
        			tail = head;
        		}else {
        			tail.next = p;
        			tail = p;
        		}
        		p = p.next;
        	}else {
        		if(head == null) {
        			head = q;
        			tail = head;
        		}else {
        			tail.next = q;
        			tail = q;
        		}
        		q = q.next;
        	}
        	
        	
        }
        while(p != null) {
        	tail.next = p;
        	tail = tail.next;
        	p = p.next;
        }
        
        while(q != null) {
        	tail.next = q;
        	tail = tail.next;
        	q = q.next;
        }
        
        tail.next = null;
        return head;
    }
    
    public static ListNode swapPairs(ListNode head) {
    	if(head == null || head.next == null) {
    		return head;
    	}
    	ListNode p = head , q = head.next , last = null;
    	while(p != null && q != null) {
    		p.next = q.next;
    		q.next = p;
    		if(last == null) {
    			head = q;
    		}else {
    			last.next = q;
    		}
    		ListNode temp = p;
    		p = q;
    		q = temp;
    		last = q;
    		p = q.next;
    		if(p != null) {
    			q = p.next;
    		}
    	}
        return head;
    }
    public static ListNode reverseKGroup(ListNode head, int k) {
    	if(head == null || head.next == null || k == 0 || k == 1) {
    		return head;
    	}
    	ListNode  p = head , last = head , t = null , q = null , temp = new ListNode(0)  , first = new ListNode(0) ;
    	temp.next = head;
    	first = temp;
    	while(last != null) {
    		for(int i = 1 ; i < k && last != null ; i++) {
    			last = last.next;
    		}
    		if(last != null) {
    			first.next = last;
    			first = p;
    			//进行局部反转
    			t = p.next; 
    			while(t != last) {
    				q = t.next;
    				t.next = p;
    				p = t;
    				t = q;
    			}
    			first.next = last.next;
    			t.next = p;
    			last = first.next;
    			p = last;
    		}
    	}

        return temp.next;
    }
    public int removeDuplicatess(int[] nums) {
        int i = 1 , j  = 0;
        for(;i < nums.length ; i++) {
        	if(nums[i] != nums[j]) {
        		nums[++j] = nums[i];
        	}
        }
        return j + 1;
    }
    public int removeElement(int[] nums, int val) {
        int i = 0 , j = -1;
        for(; i < nums.length ; i++) {
        	if(nums[i] != val) {
        		nums[++j] = nums[i];
        	}
        }
        return j + 1;
    }
    
    public static int strStr(String haystack, String needle) {
    	if(needle.equals("")) {
    		return 0;
    	}
    	int[] next = new int[needle.length()];
    	
    	caculatorNext(next, needle);
    	int i = 0 , j  = 0;
    	while(i < haystack.length()) {
    		if(haystack.charAt(i) == needle.charAt(j)) {
    			i++;
    			j++;
    		}else {
    			j = next[j];
    			if(j == -1) {
    				i++;
    				j = 0;
    			}
    		}
    		if(j == needle.length()) {
    			return i - j;
    		}
    	}
    	return -1;
    }
    
    private static void caculatorNext(int[] next , String needle) {
    	next[0] = -1;
    	int i = -1 , k = 0;
    	while(k + 1 < needle.length()) {
    		if(i == -1 || needle.charAt(i) == needle.charAt(k)) {
    			i++;
    			k++;
    			next[k] = i;
    		}else {
    			i = next[i];
    		}
    	}
    }
    
    public static int divide(int dividend, int divisor) {
        int flag = divisor > 0 ? 1 : -1;
        flag = dividend > 0 ? flag : -flag;
        long dividended = 0;
        long divisored = 0;
        if(flag == 1 && dividend == -2147483648 && divisor == -1) {
        	return -(dividend + 1);
        }
        if(dividend == -2147483648) {
        	dividended = 2147483647l + 1;
        }else {
        	dividended  = dividend > 0 ? dividend : -dividend;;
        }
        
        if(divisor == -2147483648) {
        	divisored = 2147483647l + 1;
        }else {
        	divisored  = divisor > 0 ? divisor : -divisor;;
        }
        int count = 0;
        while(dividended - divisored >= 0) {
        	long temp = dividended;
        	long rest = divisored;
        	int tempCount = 0;
        	int record = 1;
        	while((temp - rest) >= 0) {
        		temp -= rest;
        		rest <<= 1;
        		if(tempCount == 0){
        			record = 1;
        		}else {
        			record <<= 1;
        		}
        		tempCount += record;
        	}
        	count += tempCount;
        	dividended = temp;
        }
        return flag > 0 ? count : -count;
    }
    
    public static List<Integer> findSubstrings(String s, String[] words) {
    	
        List<Integer> results = new ArrayList<>();
        if(!s.equals("") && words.length  > 0) {
        	//保存数组的每个元素的初始位置
        	Map<String, List<Integer>> map = new HashMap<>();
        	for(int i = 0 ; i < words.length ; i++) {
        		List<Integer> elements = new ArrayList<>();
        		composeString(null,elements, new String[] {words[i]}, new int[] {0}, 0, "", s);
        		map.put(words[i], elements);
        	}
        	int[] used = new int[words.length];
            composeString(map,results, words, used, 0, "", s);
        }
        return results;
    }
    private static void composeString(Map<String, List<Integer>> map, List<Integer> results, String[] words, int[] used, int index, String subString, String s) {
    	if(index == words.length) {
    		List<Integer> tmp = strStrTwo(map,s, subString);
    		for(Integer element : tmp) {
    			if(!results.contains(element)) {
    				results.add(element);
    			}
    		}
    	}
    	
    	for(int i = 0 ; i < words.length ; i++) {
    		if(used[i] == 0) {
    			used[i] = 1;
    			composeString(map,results, words, used, index + 1, subString + words[i], s);
    			used[i] = 0;
    		}
    			
    	}
    }
    public static List<Integer> strStrTwo(Map<String, List<Integer>> map, String haystack, String needle) {

    	int[] next = new int[needle.length()];
    	List<Integer> list = new ArrayList<>();
    	List<Integer> elements = new ArrayList<>();
    	if(map != null) {
    		elements = map.get(needle);
    	}
    	caculatorNextTwo(next, needle);
    	int i = 0 , j  = 0 , index = 0;
    	if(map != null && elements != null && elements.size() > 0) {
    		i = 0;
        	while(i < haystack.length() && index <= elements.size()) {
        		if(haystack.charAt(i) == needle.charAt(j)) {
        			i++;
        			j++;
        		}else {
        			j = next[j];
        			if(j == -1) {
        				i = elements.get(index++);
        				j = 0;
        			}
        		}
        		if(j == needle.length()) {
        			list.add(i - j);    			
        			if(index == elements.size()) {
        				break;
        			}else {
        				i = elements.get(index++);
        			}
        			j = 0;
        		}
        	}
    	}else {
    		i = 0;
        	while(i < haystack.length()) {
        		if(haystack.charAt(i) == needle.charAt(j)) {
        			i++;
        			j++;
        		}else {
        			j = next[j];
        			if(j == -1) {
        				i++;
        				j = 0;
        			}
        		}
        		if(j == needle.length()) {
        			list.add(i - j);    			
        			i = i - j + 1;
        			j = 0;
        		}
        	}
    	}
    	return list;
    }
    private static void caculatorNextTwo(int[] next , String needle) {
    	next[0] = -1;
    	int i = -1 , k = 0;
    	while(k + 1 < needle.length()) {
    		if(i == -1 || needle.charAt(i) == needle.charAt(k)) {
    			i++;
    			k++;
    			next[k] = i;
    		}else {
    			i = next[i];
    		}
    	}
    }
    
    public static List<Integer> findSubstring(String s, String[] words) {
    	
        List<Integer> results = new ArrayList<>();
        if(!s.equals("") && words.length  > 0) {
        	//保存数组的每个元素的初始位置
        	Map<String, Integer> map = new HashMap<>();
        	//记录单词
        	for(int i = 0 ; i < words.length ; i++) {
        		map.put(words[i], map.get(words[i]) == null ? 1 : map.get(words[i]) + 1);
        	}
        	for(int i = 0 ; i < s.length() ; i++) {
        		Map<String, Integer> record = new HashMap<>();
        		int j = 0;
        		for( ; j < words.length ; j++) {
        			String temp = "";
        			if(i + words[0].length() * j + words[0].length() <= s.length()) {
        				temp = s.substring(i + words[0].length() * j, i + words[0].length() * j + words[0].length());
        			}else {
        				break;
        			}
        			if(map.containsKey(temp)) {
        				if(record.get(temp) == null) {
        					record.put(temp, 1);
        				}else if(record.get(temp) == map.get(temp)) {
        					break;
        				}else {
        					record.put(temp, record.get(temp) + 1);
        				}
        			}else {
        				break;
        			}
        		}
        		if(j == words.length) {
        			results.add(i);
        		}
        	}
        }
        return results;
    }
    
    public static void nextPermutation(int[] nums) {
    	int i = nums.length - 1; 
        for(; i > 0 ; i--) {
        	if(nums[i - 1] < nums[i]) {
        		int min = nums[i];
        		int index = i;
        		for(int j = i + 1 ; j < nums.length ; j++) {
        			if(nums[j] < min && nums[j] > nums[i - 1]) {
        				min = nums[j];
        				index = j;
        			}
        		}
        		int temp = nums[index];
        		nums[index] = nums[i - 1];
        		nums[i - 1] = temp;
        		break;
        	}
        }
        int count = i;
        for(; i < nums.length - 1 ; i++) {
        	for(int j = count ; j < nums.length - i - 1 + count; j++) {
        		if(nums[j] > nums[j + 1]) {
        			int temp = nums[j];
        			nums[j] = nums[j + 1];
        			nums[j + 1] = temp;
        		}
        	}
        }
    }
    
    public static int longestValidParentheses(String s) {
    	if(s.equals("") || s.length() == 1) {
    		return 0;
    	}
        int res = 0 , start = 0;
        Stack<Integer> stack = new Stack<>();
        for(int i = 0 ; i < s.length() ; i++) {
        	if(s.charAt(i) == '(') {
        		stack.push(i);
        	}else {
        		if(stack.isEmpty()) {
        			start = i + 1;
        		}else {
        			stack.pop();
        			res = stack.isEmpty() ? Math.max(res, i - start + 1) : Math.max(res, i - stack.peek());
        		}
        	}
        }
        return res;
    }
    public static int search1(int[] nums, int target) {
        int i = 0 , j = nums.length - 1 , mid = (i + j) / 2;
        while(i <= j) {
        	if(nums[mid] >= nums[i]) {
        		//这一段从大到小
        		if(nums[mid] < target) {
        			i = mid + 1;
        		}else if(target < nums[mid]) {
        			if(target > nums[j]) {
        				j = mid - 1;
        			}else if(target < nums[j]) {
        				if(nums[j] > nums[i]) {
        					j = mid - 1;
        				}else {
        					i = mid + 1;
        				}
        			}else {
        				return j;
        			}
        			
        		}else {
        			return mid;
        		}
        	}else {
        		//右面是从小到大
        		if(nums[mid] == target) {
        			return mid;
        		}else if(nums[mid] < target) {
        			if(target > nums[j]) {
            			j = mid - 1;
            		}else if(target < nums[j]) {
            			i = mid + 1;
            		}else {
            			return j;
            		}
        		}else {
        			j = mid - 1;
        		}
        		
        	}
        	mid = (i + j) / 2;
        }
        return -1;
    }
    
    public static  int[] searchRange(int[] nums, int target) {
        List<Integer> result = new ArrayList<>();
        findValue(nums, 0, nums.length - 1, target, result);
         int[] array = new int[] {-1,-1};
        if(result.size() > 0) {
        	array[0] = result.get(0);
        	array[1] = result.get(result.size() - 1);
        }
        
        return array;
    }
    private static void findValue(int[] nums, int start , int end , int target , List<Integer> result) {
    	if(start > end) {
    		return;
    	}   	    	    	
    	if(target < nums[start]) {
    		return;
    	}
    	
    	if(target > nums[end]) {
    		return;
    	}
    	if(target == nums[start]) {
    		result.add(start);
    	}
    	findValue(nums, start + 1, (start + end) / 2, target, result);
    	findValue(nums, (start + end) / 2 + 1, end - 1, target, result);
    	
    	if(target == nums[end]) {
    		result.add(end);
    	}
    }
    
    public static int searchInsert(int[] nums, int target) {
        int i = 0 ;
        for(;i < nums.length ; i++) {
        	if(nums[i] >= target) {
        		return i;
        	}
        }
        return i;
    }
    public static String countAndSay(int n) {
        String source = "1";
        for(int i = 1 ; i < n ; i++) {
        	String newString = "";
        	for(int j = 0 ; j < source.length() ; j++) {
        		int count = 1;
        		char cur = source.charAt(j);
        		while(j + 1 < source.length() && cur == source.charAt(j + 1)) {
        			count++;
        			j++;
        		}
        		newString += count + "" + cur;
        		
        	}
        	source = newString;
        			
        }
        return source;
    }
    
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
    	List<List<Integer>> results = new ArrayList<>();
    	List<Integer> elements = new ArrayList<>();
    	for(int i = 0 ; i < candidates.length - 1 ; i++) {
    		for(int j = 0 ; j < candidates.length - i - 1 ; j++) {
    			if(candidates[j] > candidates[j + 1]) {
    				int temp = candidates[j];
    				candidates[j] = candidates[j + 1];
    				candidates[j + 1] = temp;
    			}
    		}
    	}
    	for(int i = 0 ; i < candidates.length ; i++) {
    		elements.add(candidates[i]);
    	}

		Set<List<Integer>> set = new HashSet<>();
    	searchCompose(set , 0 , results, elements, target, 0, null);
    	return results;
    }
    
    private static void searchCompose(Set<List<Integer>> set,int index ,List<List<Integer>> results, List<Integer> candidates , int target , int sum , List<Integer> temp) {
    	if(target == sum) {
    		if(!set.contains(temp)) {
    			set.add(temp);
        		results.add(temp);
    		}
    		return;
    	}
    	if(target < sum) {
    		return;
    	}
    	for(int i = index ; i < candidates.size() ; i++) {
    			List<Integer> tmp = null;
    			if(temp == null) {
    				tmp = new ArrayList<>();
    			}else {
    				tmp = new ArrayList<>(temp);
    			}
    			 
    			tmp.add(candidates.get(i));
    			searchCompose(set ,i ,results, candidates, target, sum + candidates.get(i), tmp);
    	}
    }
    public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
    	List<List<Integer>> results = new ArrayList<>();
    	for(int i = 0 ; i < candidates.length - 1 ; i++) {
    		for(int j = 0 ; j < candidates.length - i - 1 ; j++) {
    			if(candidates[j] > candidates[j + 1]) {
    				int temp = candidates[j];
    				candidates[j] = candidates[j + 1];
    				candidates[j + 1] = temp;
    			}
    		}
    	}
    	findCompose2(0, 0, target, new ArrayList<>(), new HashSet<>(), results, candidates);
    	return results;
    }
    
    private static void findCompose2(int index , int sum , int target , List<Integer> temp , Set<List<Integer>> set,List<List<Integer>> results, int[] candidates) {
    	if(sum > target) {
    		return;
    	}
    	if(sum == target) {
    		if(!set.contains(temp)) {
    			results.add(temp);
    			set.add(temp);
    		}
    		return;
    	}
    	for(int i = index ; i < candidates.length ; i++) {
    		List<Integer> next = new ArrayList<>(temp);
    		next.add(candidates[i]);
    		if(candidates[i] + sum > target) {
    			break;
    		}else {
    			findCompose2(i + 1, sum + candidates[i], target, next, set, results, candidates);
			}
    	}
    }
    public static int firstMissingPositive(int[] nums) {
        int[] value = new int[1000];
        for(int i = 0 ; i < nums.length ; i++) {
        	if(nums[i] > 0) {
        		int index = nums[i] / 32;
            	int rest = nums[i] % 32;
            	value[index] = value[index] | (1 << rest);
        	}
        }
        int i = 0 ;
        for(; i < 1000 * 32 ; i++) {
        	int index = i / 32;
        	int rest = i % 32;
        	if((value[index] & (1 << rest)) == 0 && i > 0) {
        		break;
        	}
        }
        return i;
    }
    public static int trap(int[] height) {
        Map<Integer, Integer> numCount = new HashMap<>();
        Map<Integer, Integer> start = new HashMap<>();
        Map<Integer, Integer> end = new HashMap<>();
        for(int i = 0 ; i < height.length ; i++) {
        	for(int j = 1 ; j <= height[i] ; j++) {
        		numCount.put(j,numCount.get(j) == null ? 1 : numCount.get(j) + 1);
        		start.put(j, start.get(j) == null ? i : (start.get(j) < i ? start.get(j) : i));
        		end.put(j, end.get(j) == null ? i : (end.get(j) > i ? end.get(j) : i));
        	}
        }
        int count = 0;
        for(Entry<Integer, Integer> entry : numCount.entrySet()) {
        	if(entry.getValue() > 1) {
        		count += (end.get(entry.getKey()) - start.get(entry.getKey()) + 1 - entry.getValue());
        	}
        }
        return count;
    }
    public static int jump(int[] nums) {
    	if(nums.length == 0 || nums.length == 1) {
    		return 0;
    	}
        int[][] dp = new int[2][nums.length];
        dp[0][0] = 0;
        for(int i = 0 ; i < nums.length ; i++) {
        	int index = 1;
        	if(i > 0) {
        		index = nums[dp[1][i]] - i + dp[1][i];
        	}
        	for(int j =  index; j <= nums[i] ; j++) {
        		if((i + j) < dp[0].length && 
        				(dp[0][i + j] == 0 ||
        				dp[0][i + j] > dp[0][i] + 1)) {
        			dp[0][i + j] = dp[0][i] + 1;
        			dp[1][i + j] = i;
        		}
        	}
        }
        return dp[0][nums.length - 1];
    }
    public static List<List<Integer>> permute(int[] nums) {
    	for(int i = 0 ; i < nums.length - 1 ; i++) {
    		for(int j = 0 ; j < nums.length - i - 1 ; j++) {
    			if(nums[j] > nums[j + 1]) {
    				int temp = nums[j];
    				nums[j] = nums[j + 1];
    				nums[j + 1] = temp;
    			}
    		}
    	}
    	int[] mark = new int[nums.length];
        int[] used = new int[nums.length];
        for(int i = 1 ; i < nums.length ; i++) {
        	if(nums[i] == nums[i - 1]) {
        		mark[i] = mark[i - 1];
        	}else {
        		mark[i] = i;
        	}
        }
        List<List<Integer>> results = new ArrayList<>();
        allPL(results , new ArrayList<>(), used, nums);
        return results;
    }
    
    private static void allPL(List<List<Integer>> results , List<Integer> temp , int[] used , int[] nums) {
    	if(temp.size() == nums.length) {
    		results.add(temp);
    		return;
    	}
    	boolean flag = true;
    	int pre = 0;
    	for(int i = 0 ; i < nums.length ; i++) {
    		if(used[i] == 0) {
    			if(flag) {
    				pre = nums[i];
    				flag = false;
    			}else if(nums[i] == pre) {
    				continue;
    			}
    			pre = nums[i];
    			used[i] = 1;
    			List<Integer> next = new ArrayList<>(temp);
    			next.add(nums[i]);
    			allPL(results, next, used, nums);
    			used[i] = 0;
    		}
    	}
    }
    public static List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> record = new HashMap<>();
        for(String str : strs) {
        	char[] ch = str.toCharArray();
        	Arrays.sort(ch);
        	String mark = new String(ch);
        	if(record.get(mark) == null) {
        		List<String> list = new ArrayList<>();
        		list.add(str);
        		record.put(mark, list);
        	}else {
        		record.get(mark).add(str);
        	}
        }
        return new ArrayList<>(record.values());
    }
    
    public static double myPow(double x, int n) {
    	int flag = n > 0 ? 1 : -1;
    	double result = caculator(x, n > 0 ? n : -n);
    	if(flag == -1) {
    		result = 1.0 / result;
    	}
    	
    	return result;
    }
    
    private static double caculator(double x , int n) {
    	if(n == 0) {
    		return 1.0;
    	}else if(n == 1) {
    		return x;
    	}else if(n == 2) {
    		return x * x;
    	}else if(n % 2 == 0) {
    		double result = caculator(x, n / 2);
    		return result * result;
    	}else {
    		double result = caculator(x, n / 2);
    		return result * result * x;
    	}
    }
    public static int totalNQueens(int n) {
        return solveNQueens(n).size();
    }
    public static List<List<String>> solveNQueens(int n) {
    	List<List<String>> results = new ArrayList<>();
    	char[][] ch = new char[n][n];
    	for(int i = 0 ; i < n ; i++) {
    		for(int j = 0 ; j < n ; j++) {
    			ch[i][j] = '.';
    		}
    	}
    	findSolve(0, n, results, ch);
    	return results;
    }
    private static void findSolve(int index , int n , List<List<String>> results , char[][] ch) {
    	if(index == n) {
    		List<String> result = new ArrayList<>();
    		for(int i = 0 ; i < ch.length ; i++) {
    			result.add(new String(ch[i]));
    		}
    		results.add(result);
    		return;
    	}
    	for(int i = 0 ; i < n ; i++) {
    		//从这一行的每个位置开始判断是否符合
    		//判断上面的方向
    		boolean flag = true;
    		for(int j = index - 1 ; flag && j >= 0 ; j--) {
    			if(ch[j][i] == 'Q') {
    				flag = false;
    				break;
    			}
    		}
    		//判断左对角线
    		for(int j = index - 1 , k = i - 1 ; j >= 0 && k >= 0 && flag ; j-- , k--) {
    			if(ch[j][k] == 'Q') {
    				flag = false;
    				break;
    			}
    		}
    		
    		//右对角线
    		for(int j = index - 1 , k = i + 1 ; j >= 0 && k < n && flag ; j-- , k++) {
    			if(ch[j][k] == 'Q') {
    				flag = false;
    				break;
    			}
    		}
    		if(flag) {
    			ch[index][i] = 'Q';
    			findSolve(index + 1, n, results, ch);
    			ch[index][i] = '.';
    		}
    	}
    }
    public static int maxSubArray(int[] nums) {
        int max = 0 ;
        boolean flag = true;
        int sum = 0;
        for(int i = 0 ; i < nums.length ; i++) {
        	sum += nums[i];
        	if(flag || sum > max) {
        		max = sum;
        		flag = false;
        	}
        	if(sum < 0) {
        		sum = 0;
        	}
        }
        return max;
    }
    public static List<Integer> spiralOrder(int[][] matrix) {

        List<Integer> result = new ArrayList<>();
    	if(matrix.length == 0) {
    		return result;
    	}
        int top = 0 , bottom = matrix.length - 1 , left = 0 , right = matrix[0].length - 1;
        boolean topFlag = true , rightFlag = true , leftFlag = false , bottomFlag = false;
        int i = 0 , j = 0;
        while(!((i - 1) < top && (i + 1) > bottom && (j - 1) < left && (j + 1) > right)) {
        	//向右
        	while((j + 1) <= right && topFlag) {
        		result.add(matrix[i][j]);
        		j++;
        		rightFlag = true;
        	}
        	topFlag = false;
        	top++;
        	//向下
        	while((i + 1) <= bottom && rightFlag) {
        		result.add(matrix[i][j]);
        		i++;
        		bottomFlag = true;
        	}
        	rightFlag = false;
        	right--;
        	//向左
        	while((j - 1) >= left && bottomFlag) {
        		result.add(matrix[i][j]);
        		j--;
        		leftFlag = true;
        	}
        	bottomFlag = false;
        	bottom--;
        	//向上
        	while((i - 1) >= top && leftFlag) {
        		result.add(matrix[i][j]);
        		i--;
        		topFlag = true;
        	}
        	leftFlag = false;
        	left++;
        }
        result.add(matrix[i][j]);
        return result;
    }
    public static boolean canJump(int[] nums) {
        int[][] dp = new int[2][nums.length];
        boolean flag = true;
        int index = 1;
        for(int i = 0 ; i < nums.length ; i++) {
        	if(i != 0 && dp[0][i] == 0) {
        		flag = false;
        		break;
        	}
        	if(i != 0) {
        		index = nums[dp[1][i]] - i + dp[1][i];
        	}
        	for(int j = index ; (i + j) < nums.length && j <= nums[i] ; j++) {
        		dp[0][i + j] = dp[0][i] + 1;
        		dp[1][i + j] = i;
        	}
        }
        return flag;
    }
    public static List<Interval> merge(List<Interval> intervals) {
    	List<Interval> result = new ArrayList<>();
    	//根据start排序
    	Collections.sort(intervals, new Comparator<Interval>() {

			@Override
			public int compare(Interval o1, Interval o2) {
				if(o1.start < o2.start) {
					return -1;
				}else if(o1.start > o2.start) {
					return 1;
				}
				return 0;
			}
		});
    	Interval interval = null;
    	for(int i = 0 ; i < intervals.size() ; i++) {
    		Interval current = intervals.get(i);
    		if(interval == null) {
    			interval =  current;
    		}else {    			
    			if(current.start > interval.end) {
    				result.add(interval);
    				interval = current;
    			}else if(current.start <= interval.end) {
    				interval.end = Math.max(current.end, interval.end);
    			}
    		}
      		
    	}
    	if(interval != null) {
    		result.add(interval);
    	}
    	
    	return result;
    }
    
    public static List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        intervals.add(newInterval);
        List<Interval> result = new ArrayList<>();
    	//根据start排序
    	Collections.sort(intervals, new Comparator<Interval>() {

			@Override
			public int compare(Interval o1, Interval o2) {
				if(o1.start < o2.start) {
					return -1;
				}else if(o1.start > o2.start) {
					return 1;
				}
				return 0;
			}
		});
    	Interval interval = null;
    	for(int i = 0 ; i < intervals.size() ; i++) {
    		Interval current = intervals.get(i);
    		if(interval == null) {
    			interval =  current;
    		}else {    			
    			if(current.start > interval.end) {
    				result.add(interval);
    				interval = current;
    			}else if(current.start <= interval.end) {
    				interval.end = Math.max(current.end, interval.end);
    			}
    		}
      		
    	}
    	if(interval != null) {
    		result.add(interval);
    	}
    	
    	return result;
    }
    public static int lengthOfLastWord(String s) {
    	String[] strings = s.split(" ");
    	if(strings.length == 0) {
    		return 0;
    	}
    	return strings[strings.length - 1].length();
    }
    public static int[][] generateMatrix(int n) {
        int[][] array = new int[n][n];
        int top = 0 , bottom = n - 1 , left = 0 , right = n - 1;
        int i = 0 , j = 0 ;
        int count = 1;
        while(!((i - 1) < top && (i + 1) > bottom && (j - 1) < left && (j + 1) > right)) {
        	while(j + 1 <= right) {
        		array[i][j] = count++;
        		j++;
        	}
        	top++;
        	while(i + 1 <= bottom) {
        		array[i][j] = count++;
        		i++;
        	}
        	right--;
        	while(j - 1 >= left) {
        		array[i][j] = count++;
        		j--;
        	}
        	bottom--;
        	while(i - 1 >= top) {
        		array[i][j] = count++;
        		i--;
        	}
        	left++;
        }
        array[i][j] = count;
        return array;
    }
    public static String getPermutation(int n, int k) {
        char[] chars = new char[n];
        for(int i = 0 ; i < n ; i++) {
        	chars[i] = (char) (i + 1 + 48);
        }
        while((k--) > 1) {
        	char min = '0';
        	int index = 0;
        	for(int i = n - 1 ; i > 0 ; i--) {
        		if(chars[i] > chars[i - 1]) {
        			min = chars[i];
        			index = i;
        			for(int j = n - 1 ; j >= i ; j--) {
        				if( (chars[j] < min && chars[j] > chars[i - 1])) {
        					min = chars[j];
        					index = j;
        				}
        			}
        			chars[index] = chars[i - 1];
        			chars[i - 1] = min;
        			//排序
        			for(int j = i ; j < n - 1 ; j++) {
        				for(int t = i ; t < n - j - 1 + i; t++) {
        					if(chars[t] > chars[t + 1]) {
        						char temp = chars[t];
        						chars[t] = chars[t + 1];
        						chars[t + 1] = temp;
        					} 
        				}
        			}
        			break;
        		}
        	}
        }
        return new String(chars);
    }
    
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null || head.next == null) {
        	return head;
        }
        ListNode p = head , q = head;
        while((k--) > 0) {
        	p = p.next;
        	if(k != 0 && p.next == null) {
        		p = head;
        	}
        }
        while(p.next != null) {
        	p = p.next;
        	q = q.next;
        }
        p.next = head;
        head = q.next;
        q.next = null;
        return head;
    }
    public static int uniquePaths(int m, int n) {
    	if(m == 0 || n == 0){
            return 0;
        }
       int[][] dp = new int[m][n];
       dp[0][0] = 1;
       for(int i = 0 ; i < m  ; i++) {
    	   for(int j = 0 ; j < n ; j++) {
    		   if(j + 1 < n) {
    			   dp[i][j + 1] += dp[i][j];
    		   }
    		   if(i + 1 < m) {
    			   dp[i + 1][j] += dp[i][j];
    		   }
    	   }
       }
       return dp[m - 1][n - 1];
    }
    
    public static int uniquePathsWithObstacles(int[][] obstacleGrid) {
    	if(obstacleGrid.length == 0 || obstacleGrid[0].length == 0 || obstacleGrid[0][0] == 1) {
    		return 0;
    	}
        int[][] dp = new int[obstacleGrid.length][obstacleGrid[0].length];
        dp[0][0] = 1;
        for(int i = 0 ; i < dp.length ; i++) {
        	for(int j = 0 ; j < dp[0].length; j++) {
        		if(obstacleGrid[i][j] == 0) {
        			if(j + 1 < dp[0].length && dp[i][j + 1] == 0) {
          			   dp[i][j + 1] += dp[i][j];
          		   	}
        			if(i + 1 < dp.length && dp[i + 1][j] == 0) {
        				dp[i + 1][j] += dp[i][j];
        			}
        		}
        	
        	}
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }
    public static int minPathSum(int[][] grid) {
    	if(grid.length == 0 || grid[0].length == 0) {
    		return 0;
    	}
        int[][] dp = new int[grid.length][grid[0].length];
        for(int i = 0 ; i < grid.length ; i++) {
        	for(int j = 0 ; j < grid[0].length ; j++) {
        		if((i + 1) < grid.length && (dp[i + 1][j] == 0 || dp[i + 1][j] > dp[i][j] + grid[i][j])) {
        			dp[i + 1][j] = dp[i][j] + grid[i][j];
        		}
        		
        		if((j + 1) < grid[0].length && (dp[i][j + 1] == 0 || dp[i][j + 1] > dp[i][j] + grid[i][j])) {
        			dp[i][j + 1] = dp[i][j] + grid[i][j];
        		}
        	}
        }
        return dp[dp.length - 1][dp[0].length - 1] + grid[dp.length - 1][dp[0].length - 1] ;
    }
    public static boolean isNumber(String s) {
        try {
        	Float.parseFloat(s);
		} catch (Exception e) {
			return false;
		}
        if(s.contains("f") ||s.contains("F") || s.contains("d") || s.contains("D") || s.contains("l") || s.contains("L")) {
        	return false;
        }
        return true;
    }
    
    public static int[] plusOne(int[] digits) {
    	if(digits.length == 0){
    		return digits;
    	}
    	int value = digits[digits.length - 1] + 1;
    	if(value < 10) {
    		digits[digits.length - 1] = value;
    	}else {
    		int rest = 1;
    		for(int i = digits.length - 1; i >= 0 && rest > 0; i--) {
    			value = digits[i] + rest;
    			digits[i] = value % 10;
    			rest = value / 10;
        	}
    		if(rest != 0) {
    			int[] array = new int[digits.length + 1];
    			array[0] = rest;
    			for(int i = 1 ; i < array.length ; i++) {
    				array[i] = digits[i - 1];
    			}
    			return array;
    		}
    	}
    	
        return digits;
    }
    public static List<String> fullJustify(String[] words, int maxWidth) {
        int i = 0, j = 0;
        List<String> list = new ArrayList<>();
        while(i < words.length) {
        	int count = words[i].length();
        	int space = 0;
        	j = i;
        	while((count + space) <= maxWidth) {
        		j++;
        		if(j < words.length) {
            		space += 1;
            		count += words[j].length();
        		}else if(j >= words.length){
        			break;
        		}
        		if((count + space) > maxWidth){
        			count -= words[j].length();
        			break;
        		}
        	}
        	String string = "";
        	if(j == words.length) {
        		for(;i < j ; i++) {
            		string += words[i];
            		if(string.length() < maxWidth) {
                		string += " ";
                	}
            	}
        		if(string.length() < maxWidth) {
            		for(int k = string.length() ; k < maxWidth; k++) { 
            			string += " ";
            		}
            	}
        	}else {
        		int aveSpace = (maxWidth - count) / (j - i - 1 == 0 ? 1 : j - i - 1);
            	space = (maxWidth - count) - aveSpace * (j - i - 1 == 0 ? 1 : j - i - 1);
            	
            	for(;i < j ; i++) {
            		string += words[i];
            		for(int k = 0 ; k < aveSpace && string.length() < maxWidth; k++) { 
            			string += " ";
            		}
            		
            		if(space > 0) {
            			string += " ";
            			space--;
            		}
            	}
            	
            	if(string.length() < maxWidth && aveSpace != (maxWidth - count)) {
            		for(int k = string.length() ; k < maxWidth; k++) { 
            			string += " ";
            		}
            	}
        	}
        	
        	
        	list.add(string);
        	
        }
        return list;
    }
    
    public static int climbStairs(int n) {
        int[] dp = new int[n + 1];
        if(n == 1) {
        	return 1;
        }
        dp[1] = 1;
        dp[2] = 1;
        for(int i = 1 ; i < n; i++) {
        	if(i + 1 <= n) {
        		dp[i + 1] = dp[i + 1] + dp[i];
        	}
        	if(i + 2 <= n) {
        		dp[i + 2] = dp[i + 2] + dp[i];
        	}
        }
        return dp[n];
    }
    
    public static int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        
        for(int i = 0 ; i <= word2.length() ; i++) {
        	dp[0][i] = i;
        }
        
        for(int i = 0 ; i <= word1.length(); i++) {
        	dp[i][0] = i;
        }
        
        for(int i = 1 ; i <= word1.length() ; i++) {
        	for(int j = 1 ; j <= word2.length() ; j++) {
        		if(word1.charAt(i - 1) == word2.charAt(j - 1)) {
        			dp[i][j] = dp[i - 1][j - 1];
        		}else {
        			dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
        		}
        	}
        }
        return dp[word1.length()][word2.length()];
    }
    
    
    public static boolean searchMatrix(int[][] matrix, int target) {
        if(matrix.length == 0 || matrix[0].length == 0) {
    		return false;
    	}
    	int i = 0 , j = 0;
        for(;i < matrix.length;) {
        	if(matrix[i][0] < target) {
        		i++;
        	}else if(matrix[i][0] == target) {
        		return true;
        	}else {
        		break;
        	}
        }
        i--;
        if(i == -1) {
        	i = 0;
        }
        for(;i < matrix.length && j < matrix[0].length; ) {
        	if(matrix[i][j] < target) {
        		j++;
        	}else if(matrix[i][j] == target) {
        		return true;
        	}else {
        		return false;
        	}
        }
        
        return false;
    }
    
    public static void sortColors(int[] nums) {
        int i = 0 , j = nums.length - 1, k = 0;
        for(;k <= j && i < j;) {
        	while(nums[i] == 0) {
        		i++;
        	}
        	while(nums[j] == 2) {
        		j--;
        	}
        	if(i >= j) {
        		break;
        	}
        	if(k < i) {
        		k = i;
        	}
        	if(nums[k] == 2) {
        		nums[k] = nums[j];
        		nums[j] = 2;
        		j--;
        	}else if(nums[k] == 0) {
        		nums[k] = nums[i];
        		nums[i] = 0;
        		i++;
        		k++;
        	}else {
        		k++;
        	}
        }
    }
    public static String minWindows(String s, String t) {
        int start = -1, end = -1;
        int[] tHash = new int[128];
        for(int i = 0 ; i < t.length() ; i++) {
        	tHash[t.charAt(i)]++;
        }
        
        for(int i = 0; i < s.length() ; i++) {
        	if(tHash[s.charAt(i)] != 0) {
        		int count = 0;
            	int[] dest = new int[128];
            	for(int j = i ; j < s.length() ; j++) {
            		if(start != -1 && (end - start) <= (j - i)) {
            			break;
            		}
            		if( dest[s.charAt(j)] < tHash[s.charAt(j)]) {
            			dest[s.charAt(j)]++;
            			count++;
            			if(count == t.length()) {
            				if(start == -1 || (end - start) > (j - i)) {
            					start = i;
            					end = j;
            				}
            				break;
            			}
            		}
            	}
        	}
        }
        return start == -1 ? "" : s.substring(start, end + 1);
        
    }
    
    public static List<List<Integer>> combine(int n, int k) {
    	List<List<Integer>> results = new ArrayList<>();
    			
    	find(results, new ArrayList<>(), 0, n, k, 1);
    	return results;
    }
    
    
    public static List<List<Integer>> subsets(int[] nums) {
    	List<List<Integer>> results = new ArrayList<>();
    			
    	for(int i = 0 ; i <= nums.length ; i++) {
    		findsubsets(results, new ArrayList<>(), 0, nums, i, 0);
    	}
    	return results;
    }
    
    
    private static void findsubsets(List<List<Integer>> results , List<Integer> temp , int index , int[] nums , int k , int current) {
    	if(index == k) {
    		results.add(temp);
    		return;
    	}
    	for(int i = current ; i < nums.length ; i++) {
    		List<Integer> tem = new ArrayList<>(temp);
    		tem.add(nums[i]);
    		findsubsets(results, tem, index + 1, nums, k, i + 1);
    	}
    }
    
    
    private static void find(List<List<Integer>> results , List<Integer> temp , int index , int n , int k , int current) {
    	if(index == k) {
    		results.add(temp);
    		return;
    	}
    	for(int i = current ; i <= n ; i++) {
    		List<Integer> tem = new ArrayList<>(temp);
    		tem.add(i);
    		find(results, tem, index + 1, n, k, i + 1);
    	}
    }
    
    public static boolean exist(char[][] board, String word) {
    	if(word.equals("")) {
    		return true;
    	}
    	
    	if(board.length == 0 || board[0].length == 0) {
    		return false;
    	}
    	boolean[][] path = new boolean[board.length][board[0].length];
        for(int i = 0 ; i < board.length ; i++) {
        	for(int j = 0 ; j < board[0].length ; j++) {
        		if(board[i][j] == word.charAt(0)) {
        			path[i][j] = true;
        			if(searchWord(board, path, i, j, 1, word)) {
        				return true;
        			}
        			path[i][j] = false;
        		}
        	}
        }
        return false;
        
    }
    
    private static boolean searchWord(char[][] board,boolean[][] path, int row, int col, int index , String word) {
    	
    	if(index == word.length()) {
    		return true;
    	}
    	
    	boolean flag = false;
    	
    	char ch = word.charAt(index);
    	
    	if((row - 1) >= 0 && !path[row - 1][col] && board[row - 1][col] == ch) {
    		path[row - 1][col] = true;
    		flag = searchWord(board, path, row - 1, col, index + 1, word);
    		path[row - 1][col] = false;
    	}
    	
    	if(flag) {
    		return true;
    	}
    	
    	if((row + 1) < board.length && !path[row + 1][col] && board[row + 1][col] == ch) {
    		path[row + 1][col] = true;
    		flag = searchWord(board, path, row + 1, col, index + 1, word);
    		path[row + 1][col] = false;
    	}
    	
    	if(flag) {
    		return true;
    	}
    	
    	if((col - 1) >= 0 && !path[row][col - 1] && board[row][col - 1] == ch) {
    		path[row][col - 1] = true;
    		flag = searchWord(board, path, row, col - 1, index + 1, word);
    		path[row][col - 1] = false;
    	}
    	
    	if(flag) {
    		return true;
    	}
    	
    	if((col + 1) < board[0].length && !path[row][col + 1] && board[row][col + 1] == ch) {
    		path[row][col + 1] = true;
    		flag = searchWord(board, path, row, col + 1, index + 1, word);
    		path[row][col + 1] = false;
    	}
    	
    	if(flag) {
    		return true;
    	}
    	
    	return false;
    }
    
    
    public static int removeDuplicates(int[] nums) {
        int i = 0 , j = 0, k = 0;
        for(;j < nums.length ; j++) {
        	if(j == 0 || nums[j] != nums[j - 1]) {
        		nums[i++] = nums[j];
        		k = 1;
        	}else if(nums[j] == nums[j - 1] && k < 2) {
        		k++;
        		nums[i++] = nums[j];
        	}
        }
        return i ;
    }
    
    public static boolean search(int[] nums, int target) {
        int i = 0 , j = nums.length - 1 , mid = (i + j) / 2;
        while(i <= j) {
        	if(nums[mid] > nums[i]) {
        		//这一段从大到小
        		if(nums[mid] < target) {
        			i = mid + 1;
        		}else if(target < nums[mid]) {
        			if(target > nums[j]) {
        				j = mid - 1;
        			}else if(target < nums[j]) {
        				if(nums[j] > nums[i]) {
        					j = mid - 1;
        				}else {
        					i = mid + 1;
        				}
        			}else {
        				return true;
        			}
        			
        		}else {
        			return true;
        		}
        	}else if(nums[mid] < nums[i]){
        		//右面是从小到大
        		if(nums[mid] == target) {
        			return true;
        		}else if(nums[mid] < target) {
        			if(target > nums[j]) {
            			j = mid - 1;
            		}else if(target < nums[j]) {
            			i = mid + 1;
            		}else {
            			return true;
            		}
        		}else {
        			j = mid - 1;
        		}
        		
        	}else{
        		if(nums[mid] == target) {
        			return true;
        		}
        		if(nums[mid] > nums[j]) {
        			i = mid + 1;
        		}else if(nums[mid] < nums[j]) {
        			i = i + 1;
        		}else {

        			j = j - 1;
        		}
        		
            }
        	mid = (i + j) / 2;
        }
        return false;
    }
    
    public static ListNode deleteDuplicates(ListNode head) {
    	if(head == null || head.next == null) {
    		return head;
    	}
    	
    	ListNode p = head,q = head ;
    	while(p != null) {
    		if(p.val != q.val) {
    			q.next = p;
    		}

    		p = p.next;
    	}
    	return head;
    }

    public ListNode partition(ListNode head, int x) {
    	if(head == null || head.next == null) {
    		return head;
    	}
        ListNode bigHead = new ListNode(0);
        ListNode smallHead = new ListNode(0);
        
        ListNode p = head, r = bigHead, s = smallHead;
        
        while(p != null) {
        	if(p.val >= x) {
        		r.next = p;
        		r = p;
        	}else {
        		s.next = p;
        		s = p;
        	}
        	p = p.next;
        }
        s.next = bigHead.next;
        r.next = null;
        return smallHead.next;
        
        
    }
    
    
    public static boolean isScramble(String s1, String s2) {
    	if(s1.equals(s2)) {
    		return true;
    	}
    	return compareAndSplit(0, s1.length(), s1, s2);
    }
    
    
    private static boolean compareAndSplit(int start, int end, String s1, String s2) {
    	if(start >= end - 1) {
    		return false;
    	}
    	
    	for(int i = start + 1 ; i < end ; i++) {
    		String First = s1.substring(start, i);
        	String Second = s1.substring(i, end);
        	String newString = s1.substring(0, start) + Second + First + s1.substring(end);
        	if(s2.equals(newString)) {
        		return true;
        	}
        	boolean flag = compareAndSplit(start, i, s1, s2) == true ? true : compareAndSplit(i, end, s1, s2);
        	if(flag) {
        		return true;
        	}
        	flag = compareAndSplit(start, i, newString, s2) == true ? true : compareAndSplit(i, end, newString, s2);
        	if(flag) {
        		return true;
        	}
    	}
    	return false;
    }
    
    public int findContentChildren(int[] g, int[] s) {
        if(g.length == 0 || s.length == 0) {
        	return 0;
        }
        
        int count = 0;
        
        Arrays.sort(g);
        Arrays.sort(s);
        
        int child = 0 , candy = 0;
        while(child < g.length && candy < s.length) {
        	if(s[candy] >= g[child]) {
        		count++;
        		candy++;
        		child++;
        	}else {
        		candy++;
        	}
        }
        
        return count;
    }
    
    public  static int wiggleMaxLength(int[] nums) {
        if(nums.length == 0) {
        	return 0;
        }
        if(nums.length == 1 || (nums.length == 2 && nums[0] == nums[1])) {
        	return 1;
        }
        if(nums.length == 2) {
        	return 2;
        }
        int i = 1 , count = 1;
        int direction = 0;
        
        while(i < nums.length) {
        	if(nums[i] < nums[i - 1] && direction == 0) {
        		direction = -1;
        	}else if(nums[i] > nums[i - 1] && direction == 0) {
        		direction = 1;
        	}
        	if(nums[i] < nums[i - 1] && direction == 1) {
            	count++;
        		direction = -1;
        	}else if(nums[i] > nums[i - 1] && direction == -1) {
            	count++;
        		direction = 1;
        	}
        	i++;
        }
        if(direction != 0) {
        	count++;
        }
        return count;
    }
    
    
    
    public static String removeKdigits(String num, int k) {
       char[] stack = new char[num.length()];
       int t = 0;
        
        for(int i = 0 ; i < num.length() ; i++) {
        	if(t== 0 && num.charAt(i) != '0' || t != 0 && stack[t - 1] < num.charAt(i)) {
        		stack[t++] = num.charAt(i);
        	}else{
        		while(t != 0 && num.charAt(i) < stack[t - 1]  && k > 0) {
        			t--;
        			k--;
        		}
        		
        		if(t != 0 || (t== 0 && num.charAt(i) != '0')) {
        			stack[t++] = num.charAt(i);
        		}
        	}
        }
       
        
        while(k > 0 && t != 0) {
        	k--;
        	t--;
        }
        String result = new String(stack,0,t);
        return result.equals("") ? "0" : result;
    }
    
    public static int findMinArrowShots(int[][] points) {
        int count = 0;
        for(int i = 0 ; i < points.length ; i++) {
        	count++;
        	int end = points[i][1];
        	while(i + 1 < points.length && points[i + 1][0] <= end) {
        		
        		end = Math.min(end, points[i + 1][1]);
        		i++;
        	}
        }
        return count;
    }
    
    public static int maxProfit(int[] prices) {
        int max = 0;
        for(int i = 1 ; i < prices.length ; i++) {
        	if(prices[i] - prices[i - 1] > 0) {
        		max += prices[i] - prices[i - 1]; 
        	}
        }
        return max;
    }
    public static int canCompleteCircuit(int[] gas, int[] cost) {
    	int index = -1;
        for(int i = 0; i < gas.length ; i++) {
        	int rest = gas[i] - cost[i];
        	for(int j = (i + 1) % gas.length; j != i && rest >= 0; j = (j + 1) % gas.length) {
        		rest += (gas[j] - cost[j]);
        	}
        	
        	if(rest >= 0) {
        		index = i;
        		break;
        	}
        }
        return index;
    }
    
    public static int candy(int[] ratings) {
    	if(ratings.length == 0) {
    		return 0;
    	}
        int[] stack = new int[ratings.length];
        int[] index = new int[ratings.length];
        int t = 0;
        int min = 1;
        int i = 1 ;
        stack[t] = 1;
        index[t] = 0;
        for(; i < ratings.length ; i++) {
        	
    		if(ratings[i - 1] > ratings[i]) {
    			if(stack[i - 1] == 1) {
    				min += (i - index[i - 1] + 1);
    				stack[i - 1]++;
    				int find = i - 1;
    				while(find >= 0 && stack[index[find] - 1] != stack[find]) {
    					
    				}
    			}else {
    				min += 1;
    				index[i] = index[i - 1];
    			}
    			stack[i] = 1;
    		}else if(ratings[i - 1] < ratings[i]) {
    			index[i] = i;
    			stack[i] = stack[i - 1] + 1;
    			min += stack[i];
    		}else {
    			stack[i] = 1;
    			index[i] = i;
    			min += 1;
    		}
        }
        return min;
    }
    public static String removeDuplicateLetters(String s) {
    	int[] chs = new int[256];
    	int[] visited = new int[256];
    	char[] res = new char[s.length()];
    	int t = 0;
    	for(int i = 0 ; i < s.length(); i++) {
    		chs[s.charAt(i)]++;
    	}
    	
    	for(int i = 0; i < s.length(); i++) {
    		chs[s.charAt(i)]--;
    		if(visited[s.charAt(i)] == 1) {
    			continue;
    		}
    		
    		while(t != 0 && res[t - 1] > s.charAt(i) && chs[res[t - 1]] > 0) {
    			visited[res[t - 1]] = 0;
    			t--;
    			
    		}
    		
    		res[t++] = s.charAt(i);
    		visited[s.charAt(i)] = 1;
    	}
    	return new String(chs,0,t);
    }
    public static List<List<Integer>> subsetsWithDup(int[] nums) {
    	List<List<Integer>> results = new ArrayList<>();
    	Arrays.sort(nums);
    	for(int i = 0 ; i <= nums.length ; i++) {
    		findSubSets(nums, 0, new ArrayList<>(), results, i, 0);
    	}
    	return results;
    }
    
    private static void findSubSets(int[] nums, int current, List<Integer> temp, List<List<Integer>> results, int k, int index) {
    	if(index == k) {
    		results.add(temp);
    		return ;
    	}
    	
    	int pre = -65535;
    	boolean flag = true;
    	for(int i = current ; i < nums.length ; i++) {
    		if(flag) {
    			pre = nums[i];
    			flag = false;
    		}else {
    			if(pre == nums[i]) {
    				continue;
    			}
    			pre = nums[i];
    		}
    		List<Integer> list = new ArrayList<>(temp);
    		list.add(nums[i]);
    		findSubSets(nums, i + 1, list, results, k, index + 1);
    	}
    }
    
    
    public static List<Integer> countSmaller(int[] nums) {
    	if(nums.length == 0) {
    		return new ArrayList<>();
    	}
        Map<Integer, Integer> results = new HashMap<>();
        int[] temp = new int[nums.length];
        int[] index = new int[nums.length];
        for(int i = 0 ; i < nums.length ; i++) {
        	results.put(i, 0);
        	temp[i] = nums[i];
        	index[i] = i;
        }
        
        countsmaller(index,temp, 0, nums.length - 1, results);
        List<Integer> list = new ArrayList<>();
        for(int i = 0 ; i < nums.length ; i++) {
        	list.add(results.get(i));
        }
        
        return list;
    }
    
    private static void countsmaller(int[] index, int[] nums, int left, int right, Map<Integer, Integer> results) {
    	if(left == right) {
    		return ;
    	}
    	int mid = (left + right) / 2;
    	countsmaller(index,nums, left, mid, results);
    	countsmaller(index,nums, mid + 1, right, results);
    	
    	int i = left , j = mid + 1 , k = 0;
    	int[] next = new int[right - left + 1];
    	int[] tmp = new int[right - left + 1];
    	while(i <= mid && j <= right) {
    		if(nums[i] > nums[j]) {
    			//记录个数
    			int pre = results.get(index[i]);
    			int newVal = pre + right - j + 1;
    			results.put(index[i], newVal);
    			tmp[k] = index[i];
    			next[k++] = nums[i++];
    		}else if(nums[i] <= nums[j]){
    			tmp[k] = index[j];
    			next[k++] = nums[j++];
    		}
    	}
    	
    	while(i <= mid) {
    		tmp[k] = index[i];
    		next[k++] = nums[i++];
    		
    	}
    	
    	while(j <= right) {
    		tmp[k] = index[j];
    		next[k++] = nums[j++];
    	}
    	k = 0;
    	for(int t = left ; t <= right ; t++) {

    		index[t] = tmp[k];
    		nums[t] = next[k++];
    	}
    }
    private static ListNode newHead = null;
    public static ListNode reverseList(ListNode head) {
    	if(head == null || head.next == null) {
    		return null;
    	}
    	reverseListNode(head).next = null;
    	return newHead;
    }
    
    private static ListNode reverseListNode(ListNode node) {
    	if(node.next == null) {
    		newHead = node;
    		return node;
    	}
    	reverseListNode(node.next).next = node;
    	
    	return node;
    }
    
    public static ListNode reverseBetween(ListNode head, int m, int n) {
    	if(head == null || head.next == null) {
    		return head;
    	}
        ListNode newNode = new ListNode(0);
        
        int count = 1;
        ListNode p = head, q,t = newNode,tmp = newNode, next;
        
        //先到达开始的节点
        while(count < m) {
        	t.next = p;
        	t= t.next;
        	p = p.next;
        	count++;
        }
        tmp = p;
        t.next = null;
        
        //开始翻转;
        while(count <= n) {
        	next = p.next;
        	q = t.next;
        	t.next = p;
        	p.next = q;
        	p = next;
        	count++;
        }
        
        tmp.next = p;
        return newNode.next;
    }
    
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    	if(headA == null || headB == null) {
    		return null;
    	}
        Stack<ListNode> stackA = new Stack<>();
        Stack<ListNode> stackB = new Stack<>();
        ListNode p = headA, q = headB;
        
        while(p != null) {
        	stackA.push(p);
        	p = p.next;
        }
        
        while(q != null) {
        	stackB.push(q);
        	q = q.next;
        }
        ListNode tmp = null;
        while(!stackA.isEmpty() && !stackB.isEmpty()) {
        	p = stackA.pop();
        	q = stackB.pop();
        	if(p != q && p.next == q.next) {
        		return p.next;
        	}
        	
        	if(p == q) {
        		tmp = null;
        	}
        }
        return tmp;
    }
    
    public boolean hasCycle(ListNode head) {
    	if(head == null || head.next == null) {
    		return false;
    	}
    	ListNode kuai = head, man = head;
    	
    	while(kuai.next != null) {
    		man = man.next;
    		kuai = kuai.next.next;
    		if(man == kuai) {
    			return true;
    		}
    	}
    	
        return false;
    }
    
    public static RandomListNode copyRandomList(RandomListNode head) {
    	if(head == null) {
    		return null;
    	}
        RandomListNode newHead = new RandomListNode(0);
        
        //先复制
        RandomListNode p = head,q,newNode,copyNode = null;
        
        while(p != null) {
        	q = p.next;
        	newNode = new RandomListNode(p.label);
        	
        	p.next = newNode;
        	newNode.next = q;
        	p = q;
        }
        
        //复制指针
        p = head;
        while(p != null) {
        	copyNode = p.next;
        	if(p.random == null) {
        		copyNode.random = null;
        	}else {
        		copyNode.random = p.random.next;
        	}
        	p = copyNode.next;
        }
        
        newHead.next = head.next;
        p = head;
        //拆分节点
        while(p != null) {
        	copyNode = p.next;
        	p.next = copyNode.next;
        	p = p.next;
        	if(p != null) {
        		copyNode.next = p.next;
        	}else {
        		copyNode.next = null;
        	}
        }
        
        return newHead.next;
    }
    
    public static int calculate(String s) {
        Stack<Integer> numStack = new Stack<>();
        Stack<Character> symbolStack = new Stack<>();
        
        for(int i = 0 ; i < s.length(); i++) {
        	if(s.charAt(i) >= '0' && s.charAt(i) <= '9') {
        		numStack.push(s.charAt(i) - '0');
        		while(i + 1 < s.length() && s.charAt(i +1) >= '0' && s.charAt(i +1) <= '9') {
        			numStack.push(numStack.pop() * 10 + (s.charAt(i + 1) - '0'));
        			i++;
        		}
        	}else if(s.charAt(i) == ')') {
        		 char ch = symbolStack.pop();
        		while(ch != '(') {
        			 int v1 = numStack.pop();
            		 int v2 = numStack.pop();
            		
            		switch (ch) {
    				case '+':
    					numStack.push(v1 + v2);
    					break;
    				case '-':
    					numStack.push(v2 - v1);
    				default:
    					break;
    				}
            		ch = symbolStack.pop();
        		}
        	}else if(s.charAt(i) == ' '){
        		continue;
        	}else if((s.charAt(i) == '+' || s.charAt(i) == '-') && !symbolStack.isEmpty() && symbolStack.peek() == '-'){
        		int v1 = numStack.pop();
       		 	int v2 = numStack.pop();
       		 	char ch = symbolStack.pop();
       		 	switch (ch) {
					case '+':
						numStack.push(v1 + v2);
						break;
					case '-':
						numStack.push(v2 - v1);
					default:
						break;
					}
       		 	
       		 	symbolStack.push(s.charAt(i));
        	}else {
        		symbolStack.push(s.charAt(i));
        	}
        }
        
        while(!symbolStack.isEmpty()) {
        	 int v1 = numStack.pop();
    		 int v2 = numStack.pop();
    		 char ch = symbolStack.pop();
    		switch (ch) {
			case '+':
				numStack.push(v1 + v2);
				break;
			case '-':
				numStack.push(v2 - v1);
			default:
				break;
			}
        }
        if(numStack.isEmpty()) {
        	numStack.push(0);
        }
        
        return numStack.pop();
    }
    
    public static int findKthLargest(int[] nums, int k) {
    	if(nums.length == 0) {
    		return 0;
    	}
    	int[] array = new int[k + 1];
    	//先初始入堆
    	int i = 0 ;
    	for(; i < k ;i++) {
    		array[i + 1] = nums[i];
    		adjustBegin(array, i + 1);
    	}
    	
    	for(;i < nums.length ;i++) {
    		if(nums[i] > array[1]) {
    			array[1] = nums[i];
    			adjust(array, 1, k);
    		}
    	}
    	
    	return array[1];
    }
    
    private static void adjust(int[] array, int parent, int k) {
    	int child = parent * 2;
    	if(child > k) {
    		return;
    	}
				
		if(child + 1 > k) {
			if(array[child] < array[parent]) {
				int tmp = array[child] ;
				array[child] = array[parent];
				array[parent] = tmp;
				adjust(array, child, k);
			}
		}else {
			if(array[child] > array[child + 1] && array[child + 1] < array[parent]) {
				int tmp = array[child + 1] ;
				array[child + 1] = array[parent];
				array[parent] = tmp;
				adjust(array, child + 1, k);
			}else if(array[child] <= array[child + 1] && array[child] < array[parent]) {
				int tmp = array[child] ;
				array[child] = array[parent];
				array[parent] = tmp;
				adjust(array, child, k);
			}
		}
    
    }
    
    private static void adjustBegin(int[] array, int index) {
    	int parent = index / 2;
    	while(parent != 0 ) {
			if(array[index] < array[parent]) {
				int tmp = array[index];
				array[index] = array[parent];
				array[parent] = tmp;
			}
			index = parent;
			parent /= 2;
			
    	}
    }
    
    public static List<List<Integer>> pathSum(TreeNode root, int sum) {
    	List<List<Integer>> results = new ArrayList<>();
    	findPath(root, 0, sum, new ArrayList<>(), results);
    	return results;
    }
    
    private static void findPath(TreeNode root, int record, int sum, List<Integer> result, List<List<Integer>> results) {
    	if(root == null ) {
    		return ;
    	}
    	
    	List<Integer> temp = new ArrayList<>(result);
    	temp.add(root.val);
    	if(record + root.val == sum && root.left == null && root.right == null) {
    		results.add(temp);
    		return;
    	}
    	
    	findPath(root.left, record + root.val, sum, temp, results);
    	findPath(root.right, record + root.val, sum, temp, results);    	 	
    }
    
    public static TreeNode lowestCommonAncestors(TreeNode root, TreeNode p, TreeNode q) {
  
        Stack<TreeNode> pStack = new Stack<>();
        Stack<TreeNode> qStack = new Stack<>();
        findTreeNode(root, pStack, p);
        findTreeNode(root, qStack, q);
        while(!pStack.isEmpty() && !qStack.isEmpty()) {
        	if(pStack.size() > qStack.size()) {
        		pStack.pop();
        	}else if(pStack.size() < qStack.size()) {
        		qStack.pop();
        	}else {
        		if(qStack.peek() == pStack.peek()) {
        			return qStack.peek();
        		}
        		
        		qStack.pop();
        		pStack.pop();
        	}
        }
        return null;
    }
    
    
    private static boolean findTreeNode(TreeNode root, Stack<TreeNode> stack, TreeNode node) {
    	if(root == null) {
    		return false;
    	}
    	stack.push(root);
    	if(root == node) {
    		
    		return true;
    	}
    	
    	boolean flag = findTreeNode(root.left, stack, node);
    	if(flag) {
    		return true;
    	}
    	
    	flag = findTreeNode(root.right, stack, node);
    	if(flag) {
    		return true;
    	}else {
    		stack.pop();
    		return false;
    	}
    }
    
    public static void flatten(TreeNode root) {
        connectTreeNode(root);
    }
    
    private static void connectTreeNode(TreeNode root) {
    	if(root == null) {
    		return;
    	}
    	
    	
    	TreeNode left = root.left;
    	TreeNode right = root.right;
    	
    	
    	root.right = left;
    	root.left = null;
    	connectTreeNode(left);
    	connectTreeNode(right);
    	TreeNode p = root;
    	while(p.right != null) {
    		p = p.right;
    	}
    	
    	p.right = right;
    }
    
    public static List<Integer> rightSideView(TreeNode root) {
    	
        List<Integer> result = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        if(root != null) {
        	 queue.add(root);
        }
        searchRightView(result, queue);
        return result;
    }
    
    private static void searchRightView(List<Integer> result, Queue<TreeNode> queue) {
    	if(queue.size() == 0) {
    		return;
    	}
    	
    	Queue<TreeNode> newQueue = new ArrayDeque<>();
    	while(!queue.isEmpty()) {
    		if(queue.size() == 1) {
    			result.add(queue.peek().val);
    		}
    		
    		TreeNode root = queue.poll();
    		
    		if(root.left != null) {
    			newQueue.add(root.left);
    		}
    		
    		if(root.right != null) {
    			newQueue.add(root.right);
    		}
    	}
    	
    	searchRightView(result, newQueue);
    	
    }
    
    public static boolean canFinish(int numCourses, int[][] prerequisites) {
        int[][] relative = new int[numCourses][numCourses];
        for(int i = 0 ; i < prerequisites.length ; i++) {
        	relative[prerequisites[i][0]][prerequisites[i][1]] = 1;
        }
        int[] visited = new int[numCourses];
        int[] can = new int[numCourses];
        
        Stack<Integer> stack = new Stack<>();
        for(int i = 0; i < numCourses ; i++) {
        	if(can[i] == 0) {
        		stack.push(i);
        		visited[i] = 1;
        		if(!canfinish(numCourses, relative, stack, visited, can)) {
        			return false;
        		}
        		visited[i] = 0;
        	}
        }
        return true;
    }
    
    private static boolean canfinish(int numCourses, int[][] relative, Stack<Integer> stack, int[] visited, int[] can) {
    	
    	int course = stack.peek();
    	
    	for(int i = 0 ; i < numCourses ; i++) {
    		if(relative[course][i] == 1) {
    			if(visited[i] != 1) {
    				visited[i] = 1;
    				stack.push(i);
    				boolean flag = canfinish(numCourses, relative, stack, visited, can);
    				visited[i] = 0;
    				
    				if(!flag) {
    					return false;
    				}
    			}else {
    				return false;
    			}
    		}
    	}
    	
    	stack.pop();
    	can[course] = 1;
    	return true;
    }
    
    public static String serialize(TreeNode root) {
        StringBuffer buffer = new StringBuffer();
        serializeTreeNode(root, buffer);
        return buffer.toString();
    }
    
    private static void serializeTreeNode(TreeNode root, StringBuffer buffer) {
    	if(root == null) {
    		return ;
    	}
    	
    	
    	buffer.append(root.val + " ");
    	serializeTreeNode(root.left, buffer);
    	
    	serializeTreeNode(root.right, buffer);
    }

    // Decodes your encoded data to tree.
    public static TreeNode deserialize(String data) {
    	if("".equals(data) || data == null) {
    		return null;
    	}
    	String[] datas = data.split(" ");
        TreeNode root = new TreeNode(Integer.parseInt(datas[0]));
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        int i = 1;
        while(i < data.length()) {
        	TreeNode tmp = stack.pop();
        	int val = Integer.parseInt(datas[i]);
    		
        	if(stack.isEmpty() && val > tmp.val) {
        		TreeNode newNode = new TreeNode(val);
        		tmp.right = newNode;
        		stack.push(newNode);
        		i++;
        	}else if(val <= tmp.val) {
        		TreeNode newNode = new TreeNode(val);
        		tmp.left = newNode;
        		stack.push(tmp);
        		stack.push(newNode);
        		i++;
        	}else if(val > tmp.val && val <= stack.peek().val) {
    			TreeNode newNode = new TreeNode(val);
    			tmp.right = newNode;
        		stack.push(newNode);
        		i++;
        	}
        }
        return root;
    }
    
    public  static int longestPalindrome(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for(int i = 0 ; i < s.length() ; i++) {
        	char ch = s.charAt(i);
        	Integer val = map.get(ch);
        	if(val == null) {
        		map.put(ch, 1);
        	}else {
        		map.put(ch, val + 1);
        	}
        	
        }
        
        int count = 0;
        boolean flag = false;
        for(Integer val : map.values()) {
        	if(!flag && val % 2 == 1) {
        		flag = true;
        		count += 1;
        	}
        	
        	count += val - val % 2;
        }
        
        return count;
        
    }
    
    public static boolean wordPattern(String pattern, String str) {
    	
        String[] strings = str.split(" ");
        if(pattern.length() != strings.length) {
    		return false;
    	}
        Map<String, Character> map1 = new HashMap<>();
        Map<Character, String> map2 = new HashMap<>();
        boolean flag = true;
        int i = 0;
        for(; i < pattern.length() && i < strings.length; i++) {
        	Character ch = pattern.charAt(i);
        	Character pat = map1.get(strings[i]);
        	String string = map2.get(ch);
        	
        	if(("".equals(string) || string == null) && pat == null) {
        		map1.put(strings[i],ch);
        		map2.put(ch, strings[i]);
        	}else if(ch != pat) {
        		flag = false;
        		break;
        	}
        }
        return flag;
    } 
    
    public static List<String> findRepeatedDnaSequences(String s) {
        List<String> results = new ArrayList<>();
        Map<String, Integer> set = new HashMap<>();
        for(int i = 0 ; i + 10 < s.length(); i++) {
        	String string = s.substring(i, i + 10);
        	Integer val = set.get(string);
        	if(val == null) {
        		set.put(string,1);
        	}else if(val == 1){
        		results.add(string);
        		set.put(string, 2);
        	}
        }
         return results; 
    }
    
    public int numIslands(char[][] grid) {
        int count = 0;
        for(int i = 0 ; i < grid.length ; i++) {
        	for(int j = 0 ; j < grid[i].length ; j++) {
        		if(grid[i][j] == '1') {
        			count++;
        			descrease(grid, i, j);
        		}
        	}
        }
        return count;
    }
    
    private void descrease(char[][] grid,int i, int j) {
    	if(i - 1 >= 0 && grid[i - 1][j] == '1') {
    		grid[i - 1][j] = '0';
    		descrease(grid, i - 1, j);
    	}
    	
    	if(i + 1 < grid.length && grid[i + 1][j] == '1') {
    		grid[i + 1][j] = '0';
    		descrease(grid, i + 1, j);
    	}
    	
    	if(j - 1 >= 0 && grid[i][j - 1] == '1') {
    		grid[i][j - 1] = '0';
    		descrease(grid, i, j - 1);
    	}
    	
    	if(j + 1 < grid[0].length && grid[i][j + 1] == '1') {
    		grid[i][j + 1] = '0';
    		descrease(grid, i, j + 1);
    	}
    }
    
    
    
    public static int ladderLength(String beginWord, String endWord, List<String> wordList)     {
        Queue<Status> queue = new ArrayDeque<>();
        int min = 0;
        queue.add(new Status(beginWord, 1));
        while(!queue.isEmpty()) {
        	Status status = queue.poll();
        	if(min != 0 && min <= status.step + 1) {
        		continue;
        	}
        	String curString = status.curString;
        	for(int j = 0 ; j < wordList.size() ; j++) {
        		String str = wordList.get(j);
    			int i = 0;
            	while(i < str.length()) {
            		if(str.charAt(i) == curString.charAt(i)) {
            			i++;
            		}else if(min == 0 || min > status.step + 1){
            			String newString = curString.substring(0, i) + str.charAt(i) + curString.substring(i + 1);
            			if(newString.equals(str)) {
            				if(str.equals(endWord)) {
            					min = status.step + 1;
            				}else{
            					//添加到队列中
            					Status newStatus = new Status(str, status.step + 1);
            					wordList.remove(j);
            					j--;
            					queue.add(newStatus);
            					
            				}
            			}
            			break;
            		}else {
            			break;
            		}
            	}
    			
        	}
        }
        
        return min;
    }
    
    static class  Status{
    	String curString;
    	int step;
    	List<String> path;
    	public Status(String curString, int step) {
			this.curString = curString;
			this.step = step;
			this.path = new ArrayList<>();
		}
    	
    	public Status() {}
    	
    }
    
    public static List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
    	 List<List<String>> results = new ArrayList<>();
    	 Queue<Status> queue = new ArrayDeque<>();
    	 Map<String, Integer> record = new HashMap<>();
         int min = 0;
         Status sta = new Status(beginWord, 1);
         sta.path.add(beginWord);
         queue.add(sta);
         
         while(!queue.isEmpty()) {
         	Status status = queue.poll();
         	if(min != 0 && min < status.step + 1) {
         		continue;
         	}
         	String curString = status.curString;
         	for(int j = 0 ; j < wordList.size() ; j++) {
         		String str = wordList.get(j);
         		Integer val = record.get(str);
         		if(val == null || val >= status.step + 1) {
         			int i = 0;
                 	while(i < str.length()) {
                 		if(str.charAt(i) == curString.charAt(i)) {
                 			i++;
                 		}else if(min == 0 || min >= status.step + 1){
                 			String newString = curString.substring(0, i) + str.charAt(i) + curString.substring(i + 1);
                 			if(newString.equals(str)) {
                 				if(str.equals(endWord)) {
                 					status.path.add(str);
                 					if(min == 0 || min > status.step + 1) {
                 						min = status.step + 1;
                 						results.clear();
                 					}
                 					results.add(status.path);
                 				}else if(min == 0 || min >= status.step + 1 ){
             						//添加到队列中
                 					Status newStatus = new Status(str, status.step + 1);
                 					newStatus.path = new ArrayList<>(status.path);
                 					newStatus.path.add(str);
                 					queue.add(newStatus);
                 					if(val == null || val > status.step + 1) {
                 						record.put(str, status.step + 1);
                 					}
          					
                 				}
                 			}
                 			break;
                 		}else {
                 			break;
                 		}
                 	}
         		}else {
         			wordList.remove(j--);
         		}
     			
     			
         	}
         }
    	
    	return results;
    }

    public static boolean makesquare(int[] nums) {
        int sum = 0 ; 
        for(int i = 0 ; i < nums.length ; i++) {
        	sum += nums[i];
        }
        
        if(sum % 4 != 0 || sum == 0) {
        	return false;
        }
        
        int length = sum / 4;
        int count = 0;
        Arrays.sort(nums);
        
        while(true) {
        	if(findSum(nums, nums.length - 1, length, 0)) {
        		count++;
        	}else {
        		return false;
        	}
        	
        	if(count == 4) {
        		return true;
        	}
        }
    }
    
    private static boolean findSum(int[] nums, int index , int length, int sum) {
    	if(sum == length) {
    		return true;
    	}
    	
    	if(sum > length) {
    		return false;
    	}
    	
    	
    	for(int i = index; i >= 0; i--) {
    		
    		if(nums[i] != -1 && sum + nums[i] <= length) {
    			int tmp = nums[i];
    			nums[i] = -1;
    			if(findSum(nums,i - 1, length, sum + tmp)) {
    				return true;
    			}
    			nums[i] = tmp;
    		}
    	}
    	
    	return false;
    }
    
    static class Record{
    	int i;
    	int j;
    	int height;
    	public Record() {}
		public Record(int x, int y, int height) {
			super();
			this.i = x;
			this.j = y;
			this.height = height;
		}
    	
    }
    
    public static int trapRainWater(int[][] heightMap) {
    	if(heightMap.length == 0 || heightMap[0].length == 0) {
    		return 0;
    	}
    	Queue<Record> queue = new PriorityQueue<>(new Comparator<Record>() {
	
			@Override
			public int compare(Record o1, Record o2) {
				if(o1.height < o2.height) {
					return -1;
				}else if(o1.height > o2.height) {
					return 1;
				}
				return 0;
			}
	       }) ;
	       
       int count = 0;
       
       int[][] visited = new int[heightMap.length][heightMap[0].length];
       for(int i = 0 ; i < heightMap.length ; i++) {
    	   visited[i][0] = 1;
    	   visited[i][heightMap[0].length - 1] = 1;
    	   queue.add(new Record(i,0,heightMap[i][0]));
    	   queue.add(new Record(i,heightMap[0].length - 1,heightMap[i][heightMap[0].length - 1]));
       }
       
       for(int i = 1 ; i < heightMap[0].length - 1 ; i++) {
    	   visited[0][i] = 1;
    	   visited[heightMap.length - 1][i] = 1;
    	   queue.add(new Record(0,i,heightMap[0][i]));
    	   queue.add(new Record(heightMap.length - 1,i,heightMap[heightMap.length - 1][i]));
       }
       
       while(!queue.isEmpty()) {
    	   Record record = queue.poll();
    	   	//向上扩展
    	   	if(record.i - 1 >= 0 && visited[record.i - 1][record.j] == 0) {
    	   		if(heightMap[record.i - 1][record.j] < heightMap[record.i][record.j]) {
    	   			count += heightMap[record.i][record.j] - heightMap[record.i - 1][record.j];
    	   			heightMap[record.i - 1][record.j] = heightMap[record.i][record.j];
    	   		}
    		   queue.add(new Record(record.i - 1,record.j,heightMap[record.i - 1][record.j]));
    		   visited[record.i - 1][record.j] = 1;
    	   	}	
    	   	//向下扩展
			if(record.i + 1 < heightMap.length  && visited[record.i + 1][record.j] == 0) {
				if(heightMap[record.i + 1][record.j] < heightMap[record.i][record.j]) {
    	   			count += heightMap[record.i][record.j] - heightMap[record.i + 1][record.j];
    	   			heightMap[record.i + 1][record.j] = heightMap[record.i][record.j];
    	   		}
    		   queue.add(new Record(record.i + 1,record.j,heightMap[record.i + 1][record.j]));
    		   visited[record.i + 1][record.j] = 1;
			}
			if(record.j - 1 >= 0 && visited[record.i][record.j - 1] == 0) {
				if(heightMap[record.i][record.j - 1] < heightMap[record.i][record.j]) {
    	   			count += heightMap[record.i][record.j] - heightMap[record.i][record.j - 1];
    	   			heightMap[record.i][record.j - 1] = heightMap[record.i][record.j];
    	   		}
    		   queue.add(new Record(record.i ,record.j - 1,heightMap[record.i][record.j - 1]));
    		   visited[record.i][record.j - 1] = 1;
			}
			if(record.j + 1 < heightMap[0].length && visited[record.i][record.j + 1] == 0) {
				if(heightMap[record.i][record.j + 1] < heightMap[record.i][record.j]) {
    	   			count += heightMap[record.i][record.j] - heightMap[record.i][record.j + 1];
    	   			heightMap[record.i][record.j + 1] = heightMap[record.i][record.j];
    	   		}
    		   queue.add(new Record(record.i,record.j + 1,heightMap[record.i][record.j + 1]));
    		   visited[record.i][record.j + 1] = 1;
			}
       }
       return count;
    }
    
    public int rob(int[] nums) {
    	if(nums.length == 0) {
    		return 0;
    	}
    	int[] dp = new int[nums.length + 1];
    	dp[0] = 0;
    	dp[1] = nums[0];
    	for(int i = 1 ; i < nums.length ; i++) {
    		dp[i + 1] = Math.max(dp[i - 1] + nums[i], dp[i]);
    	}
    	return dp[nums.length];
    }
    
    public static int coinChange(int[] coins, int amount) {
        int[][] dp = new int[2][amount + 1];
        Arrays.sort(coins);
        
        for(int i = 0 ; i < coins.length ; i++) {
        	int value = coins[i];
        	
        	if(value <= amount) {
        		dp[0][value] = 1;
            	dp[1][value] = value;
            	
            	for(int j = value + 1 ; j <= amount ; j++) {
            		if((dp[1][j] == j && dp[0][j] > dp[0][j - value] + 1 && dp[1][j] == dp[1][j - value] + value) || (dp[1][j] < j)) {
            			dp[0][j] = dp[0][j - value] + 1;
                		dp[1][j] = dp[1][j - value] + value;
            		}
            	}
        	}
        	
        }
        
        return dp[1][amount] == amount ? dp[0][amount] : -1;
    }
    
    public int minimumTotal(List<List<Integer>> triangle) {
    	if(triangle.size() == 0 || triangle.get(0).size() == 0) {
    		return 0;
    	}
    	for(int i = triangle.size() - 2 ; i >= 0 ; i--) {
    		for(int j = 0 ; j < triangle.get(i).size() ; j++) {
    			triangle.get(i).set(j,triangle.get(i).get(j) +  Math.min(triangle.get(i + 1).get(j), triangle.get(i + 1).get(j + 1)));
    		}
    	}
    	
        return triangle.get(0).get(0);
    }
    
    public static int lengthOfLIS(int[] nums) {
    	if(nums.length == 0) {
    		return 0;
    	}
        int[] dp = new int[nums.length];
        int max = 1;
        for(int i = 0 ; i < nums.length ; i++) {
        	if(dp[i] == 0) {
        		dp[i] = 1;
        	}
        	
        	for(int j = i + 1 ; j < nums.length ; j++) {
        		if(nums[j] > nums[i]) {
        			if(dp[j] < dp[i] + 1) {
        				dp[j] = dp[i] + 1;
        				if(max < dp[j]) {
        					max = dp[j];
        				}
        			}
        		}
        	}
        }
        
        return max;
    }
    
    public static int calculateMinimumHP(int[][] dungeon) {
    	if(dungeon.length == 0 || dungeon[0].length == 0) {
    		return 0;
    	}
        int[][] dp = new int[dungeon.length][dungeon[0].length];
        dp[dungeon.length - 1][dungeon[0].length - 1] = 1;
        for(int i = dp.length - 1 ; i >= 0 ; i--) {
        	for(int j = dp[i].length - 1 ; j >= 0; j--) {
        		if(i - 1 >= 0) {
        			if(dp[i - 1][j] == 0) {
        					dp[i - 1][j] = dungeon[i][j] - dp[i][j] >= 0 ? 1 : dp[i][j] - dungeon[i][j];
        			}else {
        					dp[i - 1][j] = Math.min(dungeon[i][j] - dp[i][j] >= 0 ? 1 : dp[i][j] - dungeon[i][j], dp[i - 1][j]);
        			}
        		}
        		
        		if(j - 1 >= 0) {
        			if(dp[i][j - 1] == 0) {
    					dp[i][j - 1] = dungeon[i][j] - dp[i][j] >= 0 ? 1 : dp[i][j] - dungeon[i][j];
    			}else {
    					dp[i][j - 1] = Math.min(dungeon[i][j] - dp[i][j] >= 0 ? 1 : dp[i][j] - dungeon[i][j], dp[i][j - 1]);
    			}
        		}
        	}
        }
        
        return dungeon[0][0] - dp[0][0] >= 0 ? 1 : dp[0][0] - dungeon[0][0];
    }
    
  //  public static void main(String[] args) throws CloneNotSupportedException {
    //	List<String> list = new ArrayList<>();//"ted","","","","","","",""
    	/**
    	list.add("ted");
    	list.add("tex");
    	list.add("red");
    	list.add("tax");
    	list.add("tad");
    	list.add("den");
    	list.add("rex");
    	list.add("pee");**/
    	
    	//list.add("hot");
    	//list.add("dot");
    	//list.add("dog");
    	//list.add("lot");
    	//list.add("log");
    	//list.add("cog");
    	//System.out.println(findLadders("hit", "cog", list));;
    	//System.out.println(calculateMinimumHP(new int[][] {{100}}));
    	//findRepeatedDnaSequences("AAAAAAAAAAA");
    	//System.out.println(wordPattern("abba","dog cat cat fish"));
    	//TreeNode node = new TreeNode(3);
    	//node.left = new TreeNode(1);
    	//node.left.right = new TreeNode(2);
    	//node.right = new TreeNode(4);
    	//serialize(node);
    	//deserialize("3124");
    	//ListNode node = new ListNode(1);
    	//node.next = new ListNode(2);
    	//node = node.next;
    	//node.next = new ListNode(3);
    	//node = node.next;
    	//node.next = new ListNode(4);
    	//node = node.next;
    	//node.next = new ListNode(5);
    	//node = node.next;
    	//node.next = null;
		//ListNode r = reverseKGroup(head,2);
    	//int[] nums = new int[] {5,7,7,8,8,10};
    	//List<Interval> intervals = new ArrayList<>();
    	//intervals.add(new Interval(1,3));
    	//intervals.add(new Interval(2,6));
    	//intervals.add(new Interval(8,10));
    	//intervals.add(new Interval(15,18));
    	//intervals = merge(intervals);
    	//int[][] array = generateMatrix(3);
    	//System.out.println(sortColors(new int[]{}));1,7,4,9,2,5
    	//sortColors(new int[]{1,0,2});"222222222222222222222210"
    	
    	
    	//RandomListNode randomNode1 = new RandomListNode(1);
    	
    	//RandomListNode randomNode2 = new RandomListNode(2);
    	
    	//RandomListNode randomNode3 = new RandomListNode(3);
    	//RandomListNode randomNode4 = new RandomListNode(4);
    	
    	
    	//randomNode1.next = randomNode2;
    	
    	//randomNode2.next = randomNode3;
    	
    	//randomNode3.next = randomNode4;
    	
    	//randomNode4.next = null;
    	
    	//randomNode1.random = randomNode3;
    	//randomNode2.random = randomNode1;
    	//randomNode4.random = randomNode2;
    	
    	//System.out.println(copyRandomList(randomNode1));
    	
    	/*MyStack stack = new MyStack();
    	stack.push(1);
    	stack.push(2);
    	stack.push(3);
    	stack.push(4);
    	System.out.println(stack.pop());
    	System.out.println(stack.pop());
    	System.out.println(stack.pop());
    	System.out.println(stack.pop());*/
    	
    	//System.out.println(findKthLargest(new int[] {2},1));
    	
    	//Person person = new Person("zhangsan");
    	//Person person2 = (Person) person.clone();
    	
    	//System.out.println(person.name);
    	//System.out.println(person2.name);
    	
    	//person.name.append("jjj");
    	//System.out.println(person.name);
    	//System.out.println(person2.name);
    	
    	
    	//TreeNode root = new TreeNode(3);
    	//TreeNode left = new TreeNode(5);
    	//TreeNode right = new TreeNode(1);
    	//root.right = right;
    	//root.left = left;
    	//rightSideView(root);
    	//canFinish(2,new int[][] {{1,0}});
    	
   // }   
    
    public int numJewelsInStones(String J, String S) {
        boolean[] chs = new boolean[255];
        for(int i = 0 ; i < J.length() ; i++) {
        	chs[J.charAt(i)] = true;
        }
        int ans = 0;
        for(int i = 0 ; i < S.length() ; i++) {
        	if(chs[S.charAt(i)]) {
        		ans++;
        	}
        }
        
        return ans;
    }
    
    public List<List<Integer>> levelOrder(TreeNode root) {
    	if(root == null) {
    		return new ArrayList<>();
    	}
    	List<List<Integer>> results = new ArrayList<>();
    	List<Integer> result = new ArrayList<>();
    	TreeNode last = root;
    	Queue<TreeNode> queue = new ArrayDeque<>();
    	
    	queue.add(root);
    	
    	TreeNode curNode = null;
    	TreeNode nlast = null;
    	while(!queue.isEmpty()) {
    		curNode = queue.poll();
    		result.add(curNode.val);
    		if(curNode.left != null) {
    			queue.add(curNode.left);
    			nlast = curNode.left;
    		}
    		
    		if(curNode.right != null) {
    			queue.add(curNode.right);
    			nlast = curNode.right;
    		}
    		if(curNode == last) {
    			results.add(result);
    			result = new ArrayList<>();
    			last = nlast;
    		}
    	}
    	return results;
    }
    
    public List<Integer> preorderTraversal(TreeNode root) {
    	if(root == null) {
    		return new ArrayList<>();
    	}
    	
    	Stack<TreeNode> stack = new Stack<>();
    	List<Integer> result = new ArrayList<>();
    	
    	stack.push(root);
    	while(!stack.isEmpty()) {
    		TreeNode node = stack.pop();
    		
    		result.add(node.val);
    		
    		if(node.right != null) {
    			stack.push(node.right);
    		}
    		
    		if(node.left != null) {
    			stack.push(node.left);
    		}
    	}
    	
    	return result;
    	
    }
    
    public List<Integer> postorderTraversal(TreeNode root) {
        if(root == null) {
        	return new ArrayList<>();
        }
        
        Stack<TreeNode> stack1 = new Stack<>();
        Stack<TreeNode> stack2 = new Stack<>();
        
        stack1.push(root);
        List<Integer> result = new ArrayList<>();
        
        while(!stack1.isEmpty()) {
        	TreeNode node = stack1.pop();
        	
        	stack2.push(node);
        	if(node.left != null) {
        		stack1.push(node.left);
        	}
        	
        	if(node.right != null) {
        		stack1.push(node.right);
        	}
        }
        
        
        while(!stack2.isEmpty()) {
        	result.add(stack2.pop().val);
        }
        
        return result;
    }
    
    public boolean isBalanced(TreeNode root) {
    	if(root == null) {
    		return true;
    	}
        boolean flag = true;
        
        flag = isBalanced(root.left);
        if(flag) {
        	flag = isBalanced(root.right);
        }
        
        if(flag) {
        	flag = Math.abs(height(root.left) - height(root.right)) <= 1;
        }
        
        return flag;
    }
    
    private int height(TreeNode root) {
    	if(root == null) {
    		return 0;
    	}
    	
    	int left = height(root.left);
    	int right = height(root.right);
    	
    	return Math.max(left, right) + 1;
    }
    
    public int minDeletionSize(String[] A) {
    	int count = 0;
    	if(A.length == 0 || A[0].length() == 0) {
    		return 0;
    	}
    	
    	for(int i = 0 ;  i < A[0].length() ; i++) {
    		char pre = 'a' - 1;
    		for(int j = 0 ; j < A.length ; j++) {
    			if(pre < A[j].charAt(i)) {
    				pre = A[j].charAt(i);
    			}else {
    				count++;
    				break;
    			}
    		}
    	}
    	
    	return count;
    }
    
    public boolean lemonadeChange(int[] bills) {
        
    	if(bills.length == 0) {
    		return true;
    	}
    	
    	int[] array = new int[11];
    	
    	Arrays.sort(bills);
    	
    	for(int i = 0 ; i < bills.length ; i++) {
    		if(bills[i] == 5) {
    			array[5]++;
    		}
    		
    		if(bills[i] == 10) {
    			if(array[5] >= 1) {
    				array[5]--;
    				array[10]++;
    			}else {
    				return false;
    			}
    		}
    		
    		if(bills[i] == 20) {
    			if(array[5] >= 1 && array[10] >= 1) {
    				array[5]--;
    				array[10]--;
    			}else if(array[5] >= 3) {
    				array[5] -= 3;
    			}else {
    				return false;
				}
    		}
    		
    		
    	}
    	return true;
    }
    public List<Integer> partitionLabels(String S) {
    	List<Integer> result = new ArrayList<>();
        int[] array = new int[26];
        for(int i = 0 ; i < S.length() ; i++) {
        	array[S.charAt(i) - 'a']++;
        }
        
        int[] clone = Arrays.copyOf(array, array.length);
        int start = 0;
        for(int i = 0 ; i < S.length() ; i++) {
        	clone[S.charAt(i) - 'a']--;
        	//判断
        	boolean flag = true;
        	for(int j = 0 ; j < 26 ; j++) {
        		if(!(clone[j] == array[j] || clone[j] == 0)) {
        			flag = false;
        			break;
        		}
        	}
        	
        	if(flag) {
        		result.add(i - start + 1);
        		start = i + 1;
        		clone = Arrays.copyOf(array, array.length);
        	}
        }
        
        return result;
    }
    
    public int[] sortedSquares(int[] A) {
    	int[] result = new int[A.length];
        for(int i = 0 ; i < A.length ; i++) {
        	result[i] = A[i] * A[i];
        }
        Arrays.sort(result);
        return result;
    }
    public List<String> letterCasePermutation(String S) {
    	List<String> result = new ArrayList<>();
    	letter(result, S, 0, "");
    	return result;
    }
    private void letter(List<String> result, String S, int index, String temp) {
    	if(index == S.length() - 1) {
    		result.add(temp);
    		return;
    	}
    	char ch = S.charAt(index);
    	if(ch >= '0' && ch <= '9') {
    		letter(result, S, index + 1, temp + ch);
    	}else {
    		letter(result, S, index + 1, temp + Character.toUpperCase(ch));
    		letter(result, S, index + 1, temp + Character.toLowerCase(ch));
    	}  	
    }
    static int min = -1;
    public int minCut(String s) {
    	partition(s);
    	return min;
    } 
    
    public static List<List<String>> partition(String s) {
    	List<List<String>> result = new ArrayList<>();
    	if(!(s == null || "".equals(s))) {
    		boolean[][] huiwen = new boolean[s.length()][s.length()];
        	for(int i = 0 ; i < s.length() ; i++) {
        		for(int j = i ; j >= 0 ; j--) {
        			if(i == j || (i == j + 1 && s.charAt(i) == s.charAt(j)) || (huiwen[j + 1][i - 1] && s.charAt(i) == s.charAt(j))) {
        				huiwen[j][i] = true;
        			}
        		}
        	}
        	execute(result, s, huiwen, 0, new ArrayList<>());
    	}
    	return result;
    }
    
    private static void execute(List<List<String>> result,String s , boolean[][] huiwen, int start, List<String> temp) {
    	if(min != -1 && temp.size() - 1 >= min) {
    		return ;
    	}
    	if(start == huiwen[0].length) {
    		if(min == -1 || temp.size() - 1 < min) {
    			min = temp.size() - 1;
    		}
    		return;
    	}
    	
    	for(int i = start; i < huiwen.length ; i++) {
    		if(huiwen[start][i]) {
    			List<String> next = new ArrayList<>(temp);
    			next.add(s.substring(start, i + 1));
    			execute(result, s, huiwen, i + 1, next);
    		}
    	}
    }
	public int maxProfitss(int[] prices) {
        int minprice = Integer.MAX_VALUE;
        int profit = 0;
        for(int i = 0 ; i < prices.length ; i++) {
        	if(prices[i] < minprice) {
        		minprice = prices[i];
        	}else if(prices[i] - minprice > profit) {
        		profit = prices[i] - minprice;
        	}
        }
        return profit;
    }
	
	public int maxProfit(int k, int[] prices) {
		if(prices.length == 0) {
			return 0;
		}
		if(k > prices.length / 2) {
			int minprice = Integer.MAX_VALUE;
	        int profit = 0;
	        for(int i = 0 ; i < prices.length ; i++) {
	        	if(prices[i] < minprice) {
	        		minprice = prices[i];
	        	}else if(prices[i] - minprice > profit) {
	        		profit = prices[i] - minprice;
	        	}
	        }
	        return profit;
		}else {

	        int[][] local = new int[prices.length][k + 1];
	        int[][] gloal = new int[prices.length][k + 1];
	        
	        for(int i = 1 ; i < prices.length ; i++) {
	        	int diff = prices[i] - prices[i - 1];
	        	for(int j = 1 ; j < k + 1 ; j++) {
	        		local[i][j] = Math.max(gloal[i - 1][j - 1] + Math.max(diff, 0), local[i - 1][j] + diff);
	        		gloal[i][j] = Math.max(local[i][j], gloal[i - 1][j]);
	        	}
	        }
	        return gloal[prices.length - 1][k];
		}
    }
	
	public static boolean canPartitionKSubsets(int[] nums, int k) {
		Arrays.sort(nums);
		int sum = 0 ; 
		for(int p = 0 ; p < nums.length ; p++) {
			sum += nums[p];
		}
		
		int ave = sum / k;
		if(sum % k != 0) {
			return false;
		}
		return dfscanPartitionKSubsets(nums, new boolean[nums.length], 0, ave, 0);
		
    }
	
	public static boolean dfscanPartitionKSubsets(int[] nums, boolean[] used, int sum, int target, int index) {
		if(sum == target) {
			return true;
		}else if(sum > target) {
			return false;
		}
		
		
		for(int i = nums.length - 1 ; i >= 0 ; i--) {
			if(!used[i]) {
				used[i] = true;
				if(!dfscanPartitionKSubsets(nums, used, sum + nums[i], target, index + 1)) {
					used[i] = false;
					if(index == 0) {
						return false;
					}
				}else if(index != 0){
					return true;
				}
			}
		}
		return false;
	}
	
	public int maxProfit(int[] prices, int fee) {
		boolean flag = false;
		int sum = 0;
        for(int i = 1 ; i < prices.length ; i++) {
        	if(!flag) {
        		if(prices[i] - prices[i - 1] - fee >= 0) {
        			flag = true;
        			sum += prices[i] - prices[i - 1] - fee;
        		}
        	}else {
        		if(prices[i] - prices[i - 1] >= 0) {
        			flag = true;
        			sum += prices[i] - prices[i - 1];
        		}else {
        			flag = false;
        		}
        	}
        }
        return sum;
    }
	
	public int maxProfitss(int[] prices, int fee) {
        int[] dp1 = new int[prices.length];
        int[] dp2 = new int[prices.length];
        
        dp1[0] = -prices[0];
        
        for(int i = 1 ; i < prices.length ; i++) {
        	dp1[i] = Math.max(dp1[i - 1], dp2[i] - prices[i]);
        	dp2[i] = Math.max(dp2[i - 1], prices[i] - dp1[i] - fee);
        }
        
        return dp2[prices.length - 1];
        
    }
	
	public int findMin(int[] nums) {
		if(nums.length == 0) {
			return 0;
		}
		for(int i = 1 ; i < nums.length ; i++) {
			if(nums[i] < nums[i - 1]) {
				return nums[i];
			}
		}
        return nums[0];
    }
	
	public int minCostClimbingStairs(int[] cost) {
        int[] dp = new int[cost.length + 1];
        if(cost.length == 0) {
        	return 0;
        }
        if(cost.length == 1) {
        	return cost[0];
        }
  
        
        dp[0] = cost[0];
        dp[1] = cost[1];
        
        for(int i = 2 ; i < cost.length ; i++) {
        	dp[i] = Math.min(dp[i - 2] + cost[i], dp[i - 1] + cost[i]);
        }
        return Math.min(dp[cost.length - 1], dp[cost.length - 2]);
    }
	
	public int[] countBits(int num) {
		int[] result = new int[num + 1];
		result[0] = 0;
        int count = 1;
        for(int i = 1 ; i <= num ; i++) {
        	result[i] = 1 + result[i - count];
        	if(count * 2 == i + 1) {
        		count *= 2;
        	}
        }
        return result;
    }
	
	public int hammingWeight(int n) {
        int count = 0;
        int m = n;
		while(m > 0) {
        	if(m % 2 != 0) {
        		count++;
        	}
        	m /= 2;
        }
		return count;
    }
	
	public int hammingDistance(int x, int y) {
        return hammingWeight(x ^ y);
    }
	
	public int totalHammingDistance(int[] nums) {
		int total = 0;
       for(int i = 0 ; i < nums.length ; i++) {
    	   for(int j = i ; j < nums.length ; j++) {
    		   total += hammingDistance(nums[i], nums[j]);
    	   }
       }
       return total;
    }
	
	public boolean stoneGame(int[] piles) {
		int[][] dp = new int[piles.length][piles.length];
		for(int i = 0 ; i < piles.length ; i++) {
			dp[i][i] = piles[i];
		}
		
		for(int dis = 1 ; dis < piles.length ; dis++) {
			for(int i = 0 ; i < piles.length - dis ; i++) {
				dp[i][i + dis] = Math.max(piles[i] - dp[i + 1][i + dis], piles[i + dis] - dp[i][i + dis - 1]);
			}
		}
		return dp[0][piles.length - 1] > 0;
    }
	
	public int numTrees(int n) {
		int[] dp = new int[n + 1];
		dp[0] = 1;
		for(int i = 1 ; i <= n ; i++) {
			for(int j = 1 ; j <= i ; j++) {
				dp[i] += dp[j - 1] * dp[i - j];
			}
		}
		return dp[n]; 
    }
	
	public int numberOfArithmeticSlices(int[] A) {
		int count = 0;
		boolean[][] dp = new boolean[A.length][A.length];
		for(int i = 0 ; i < A.length ; i++) {
			for(int j = i ; j >= 0 ; j--) {
				if(i - j == 2 && A[i] - A[i - 1] == A[j + 1] - A[j]) {
					dp[j][i] = true;
					count++;
				}else if(i - j > 2 && dp[j + 1][i] && A[j + 1] - A[j] == A[j + 2] - A[j + 1]){
					dp[j][i] = true;
					count++;
				}
			}
		}
		return count;
    }
	public int distributeCandies(int[] candies) {
        Arrays.sort(candies);
        int ave = candies.length / 2;
        int count = 1;
        for(int i = 1 ; i < candies.length ; i++) {
        	if(candies[i] != candies[i - 1]) {
        		count++;
        	}
        }
        if(count > ave) {
        	return ave;
        }else {
        	return count;
        }
    }
	
	public String[] uncommonFromSentences(String A, String B) {
		String[] AStr = A.split(" ");
		String[] BStr = B.split(" ");
		Map<String, Integer> map = new HashMap<>();
		for(int i = 0 ; i < A.length() ; i++) {
			map.put(AStr[i], map.get(AStr[i]) == null ? 1 : map.get(AStr[i]) + 1);
		}
		for(int i = 0 ; i < B.length() ; i++) {
			map.put(BStr[i], map.get(BStr[i]) == null ? 1 : map.get(BStr[i]) + 1);
		}
		
		List<String> list = new ArrayList<>();
		
		for(Entry<String, Integer> entry : map.entrySet()) {
			if(entry.getValue() == 1) {
				list.add(entry.getKey());
			}
		}
		String[] result = new String[list.size()];
		return list.toArray(result);
    }

	public int singleNumber(int[] nums) {
		int result = nums[0];
		
		for(int i = 1 ; i < nums.length ; i++) {
			result ^= nums[i];
		}
		
		return result;
    }

	public int missingNumber(int[] nums) {
        int sum = ((0 + nums.length) * nums.length) / 2;
        for(int i = 0 ; i < nums.length ; i++) {
        	sum -= nums[i];
        }
        
        return sum;
    }
	
	public int minSwapsCouples(int[] row) {
		int[] array = new int[row.length];
		for(int i = 0 ; i < row.length ; i++) {
			array[row[i]] = i;
		}
		int count = 0;
		for(int i = 0 ; i < row.length - 1 ;) {
			int couple = row[i] % 2 == 0 ? row[i] + 1 : row[i] - 1;
			if(row[i + 1] != couple) {
				int couple_pos = array[couple];
				int tmp = row[i + 1];
				row[i + 1] = couple;
				row[couple_pos] = tmp;
				array[couple] = i + 1;
				array[tmp] = couple_pos;
				count++;
			}
			
			i += 2;
		}
		return count;
    }
	
	public void deleteNode(ListNode node) {
		ListNode p = node;
		while(p.next.next != null) {
			p.val = p.next.val;
			p = p.next;
		}
		
		p.val = p.next.val;
		p.next = null;
    }
	public int maxDepth(TreeNode root) {
        if(root == null) {
        	return 0;
        }
        
        int maxLeft = maxDepth(root.left);
        int maxRight = maxDepth(root.right);
        return Math.max(maxLeft, maxRight) + 1;
    }
	
	public void reverseString(char[] s, int start, int end) {
		int i = start, j = end;
		while(i < j) {
			char tmp = s[i];
			s[i] = s[j];
			s[j] = tmp;
			i++;
			j--;
		}
		
    }
	
	public String reverseWords(String s) {
		char[] result = s.toCharArray();
		int i = 0 ;
        for(; i < result.length ;) {
        	for(int j = i + 1; j < result.length ; j++) {
        		if(result[j] == ' ') {
        			reverseString(result, i, j - 1);
        			i = j + 1;
        			break;
        		}
        	}
        }
        reverseString(result, i, result.length - 1);
        return new String(result);
    }
	static int maxPathSumValue = 0;
	public static int maxPathSum(TreeNode root) {
		maxPathSumValue = Integer.MIN_VALUE;
		int tmp = findMaxPathSum(root);
		return maxPathSumValue == 0 ? tmp : maxPathSumValue;
    }
	
	public static int findMaxPathSum(TreeNode root) {
		if(root == null) {
			return 0;
		}
		
		int left = findMaxPathSum(root.left);
		int right = findMaxPathSum(root.right);
		if(left + right + root.val > maxPathSumValue) {
			maxPathSumValue = left + right + root.val;
		}
		if(right + root.val > maxPathSumValue) {
			maxPathSumValue = right + root.val;
		}
		if(left + root.val > maxPathSumValue) {
			maxPathSumValue = left + root.val;
		}
		if(root.val > maxPathSumValue) {
			maxPathSumValue = root.val;
		}
		if(left > 0 && right > 0) {
			return Math.max(left + root.val, right + root.val);
		}
		
		if(left > 0) {
			return left + root.val;
		}
		
		if(right > 0) {
			return right + root.val;
		}
		
		return root.val;
		
	}
	
	
	 public int largestRectangleArea(int[] heights) {
	        Stack<Integer> stack = new Stack<>();
	        int max = 0;
	        for(int i = 0 ; i < heights.length ; i++) {
	        	if(stack.isEmpty() || stack.peek() <= heights[i]) {
	        		stack.push(heights[i]);
	        	}else {
	        		int count = 0;
	        		while(!stack.isEmpty() && stack.peek() > heights[i]) {
	        			count++;
	        			if(max < stack.peek() * count) {
	        				max = stack.peek() * count;
	        			}
	        			stack.pop();
	        		}
	        		for(int j = 0 ; j <= count ; j++) {
	        			stack.push(heights[i]);
	        		}
	        	}
	        }
	        
	        
	        int count = 0;
	        while(!stack.isEmpty()) {
	        	count++;
				if(max < stack.peek() * count) {
					max = stack.peek() * count;
				}
	            stack.pop();
	        }
	        return max;
	    }
	public int maximalRectangle(char[][] matrix) {
		if(matrix.length == 0 || matrix[0].length == 0) {
			return 0;
		}
        int[][] heights = new int[matrix.length][matrix[0].length];
        int max = 0;
        for(int j = 0 ; j < matrix[0].length ; j++) {
        	int count = 0;
        	for(int i = 0 ; i < matrix.length ; i++) {
        		if(matrix[i][j] == '1') {
        			count++;
        		}else {
        			count = 0;
        		}
        		heights[i][j] = count;
        	}
        }
        
        for(int i = 0 ; i < matrix.length ; i++) {
        	int tmp = largestRectangleArea(heights[i]);
        	if(tmp > max) {
        		max = tmp;
        	}
        }
        return max;
    }
	
	public int[] nextGreaterElement(int[] nums1, int[] nums2) {
		Map<Integer, Integer> map = new HashMap<>();
		Stack<Integer> stack = new Stack<>();
		for(int i = 0 ; i < nums2.length ; i++) {
			if(stack.isEmpty() || stack.peek() > nums2[i]) {
				stack.push(nums2[i]);
			}else {
				while(!stack.isEmpty() && stack.peek() <= nums2[i]) {
					map.put(stack.pop(), nums2[i]);
				}
				stack.push(nums2[i]);
			}
		}
		int[] result = new int[nums1.length];
		for(int i = 0 ; i < nums1.length ; i++) {
			result[i] = map.get(nums1[i]) == null ? -1 : map.get(nums1[i]);
		}
		return result;
    }
	public int[] nextGreaterElements(int[] nums) {
		Stack<Integer> stack = new Stack<>();
		Stack<Integer> index = new Stack<>();
		int[] result = new int[nums.length];
		Arrays.fill(result, -1);
		for(int i = 0 ; i < nums.length ; i++) {
			if(stack.isEmpty() || stack.peek() >= nums[i]) {
				stack.push(nums[i]);
				index.push(i);
			}else {
				while(!stack.isEmpty() && stack.peek() < nums[i]) {
					stack.pop();
					result[index.pop()] = nums[i];
				}
				stack.push(nums[i]);
				index.push(i);
			}
		}
		
		for(int i = 0 ; i < nums.length - 1 ; i++) {
			if(stack.isEmpty() || stack.peek() >= nums[i]) {
				stack.push(nums[i]);
				index.push(i);
			}else {
				while(!stack.isEmpty() && stack.peek() < nums[i]) {
					stack.pop();
					result[index.pop()] = nums[i];
				}
				stack.push(nums[i]);
				index.push(i);
			}
		}
		return result;
	}
	
	public int minMoves(int[] nums) {
		int min = Integer.MAX_VALUE;
		int sum = 0;
		for(int i = 0 ; i < nums.length ; i++) {
			if(min > nums[i]) {
				min = nums[i];
			}
			
			sum += nums[i];
		}
		
		return sum - min * nums.length;
    }
	
	
	public int minMoves2(int[] nums) {
		int sum = 0;
		for(int i = 0 ; i < nums.length ; i++) {
			sum += nums[i];
		}
		
		int ave = sum / nums.length;
		int min = 0;
		for(int i = 0 ; i < nums.length ; i++) {
			min += Math.abs(nums[i] - ave);
		}
		return min;
    }
	
	
	public List<Integer> inorderTraversal(TreeNode root) {
		List<Integer> result = new ArrayList<>();
		Stack<TreeNode> stack = new Stack<>();
		TreeNode p = root;
		while(!stack.isEmpty() || p != null) {
			if(p == null) {
				p = stack.pop();
				result.add(p.val);
				p = p.right;
			}
			while(p != null) {
				stack.push(p);
				p = p.left;
			}
		}
		return result;
    }
	
	public int minAddToMakeValid(String S) {
		int count = 0;
		Stack<Character> stack = new Stack<>();
		for(int i = 0 ; i < S.length() ; i++) {
			char ch = S.charAt(i);
			if(ch == '(') {
				stack.push(ch);
			}else if(ch ==')') {
				if(stack.isEmpty()) {
					count++;
				}else {
					stack.pop();
				}
			}
		}
		
		return count + stack.size();
    }
	
	public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
		int flag = 1;
		List<List<Integer>> result = new ArrayList<>();
		TreeNode lastNode = root;
		TreeNode currentNode = root;
		TreeNode nextNode = null;
		
		
		return result;
    }
	
	public int[] dailyTemperatures(int[] T) {
		Stack<Integer> stack1 = new Stack<>();
		Stack<Integer> stack2 = new Stack<>();
		
		int[] result = new int[T.length];
		for(int i = 0 ; i < T.length ; i++) {
			if(stack1.isEmpty() || stack1.peek() >= T[i]) {
				stack1.push(T[i]);
				stack2.push(i);
			}else {
				while(stack1.peek() < T[i]) {
					stack1.pop();
					int index = stack2.pop();
					result[index] = i - index;
				}
			}
		}
		return result;
    }
	public int[] intersection(int[] nums1, int[] nums2) {
		Set<Integer> set1 = new HashSet<>();
		Set<Integer> set2 = new HashSet<>();
		for(int i = 0 ; i < nums1.length ; i++) {
        	set1.add(nums1[i]);
        }
		
		for(int i = 0 ; i < nums2.length ; i++) {
        	set2.add(nums2[i]);
        }
        int[] array = new int[Math.min(nums1.length, nums2.length)];
        int index = 0;
        for(Integer value : set2) {
        	if(set1.contains(value)) {
        		array[index++] = value;
        	}
        }
        return Arrays.copyOfRange(array, 0, index);
    }
	
	
	public int[] intersect(int[] nums1, int[] nums2) {
		Arrays.sort(nums1);
		Arrays.sort(nums2);
		int[] array = new int[Math.min(nums1.length, nums2.length)];
        int index = 0;
        int i = 0 , j = 0;
        while(i < nums1.length && j < nums2.length) {
        	if(nums1[i] == nums2[j]) {
        		array[index++] = nums1[i];
        		i++;
        		j++;
        	}else if(nums1[i] > nums2[j]) {
        		j++;
        	}else {
        		i++;
        	}
        }
        return Arrays.copyOfRange(array, 0, index);
    }
	
	public static String strWithout3a3b(int A, int B) {
        char[] strings = new char[A + B];
        int ALen = 0, BLen = 0;
        for(int i = 0 ; i < strings.length ; i++) {
        	if((A > B && ALen < 2) || (A < B && BLen == 2)) {
        		ALen++;
        		A--;
        		BLen = 0;
        		strings[i] = 'a';
        	}else if((A > B && ALen == 2) || (A < B && BLen < 2)) {
        		BLen++;
        		B--;
        		ALen = 0;
        		strings[i] = 'b';
        	}else if(A == B && (i == 0 || strings[i - 1] == 'a')) {
        		BLen++;
        		B--;
        		ALen = 0;
        		strings[i] = 'b';
        	}else if(A == B && strings[i - 1] == 'b') {
        		ALen++;
        		A--;
        		BLen = 0;
        		strings[i] = 'a';
        	}
        }
        return new String(strings);
    }
	
	class Word{
		String word;
		Integer count;
		public Word(String word, Integer count) {
			this.word = word;
			this.count = count;
		}
	}
	
	
	public static List<Integer> splitIntoFibonacci(String S) {
		List<Integer> result = new ArrayList<>();
		findSplitIntoFibonacci(result, 0, 1, S);
		return result;
	}
	
	public static boolean findSplitIntoFibonacci(List<Integer> result, int index, int count, String string) {
		if(string.length() == 0) {
			if(result.size() >= 3) {
				return true;
			}else {
				return false;
			}
		}
		boolean flag = false;
		for(int i = 1 ; i <= string.length() ; i++) {
			if(count > 10) {
				return false;
			}
			Long tmpvalue = Long.parseLong(string.substring(0, i));
			if(tmpvalue > Integer.MAX_VALUE) {
				return false;
			}
			
			Integer value = Integer.parseInt(string.substring(0, i));
			if(i != 1 && value == Integer.parseInt(string.substring(1, i))) {
				return false;
			}
			if(index == 0 || index == 1) {
				result.add(value);
			}else if(result.get(index - 1) + result.get(index - 2) == value){
				result.add(value);
			}else {
				continue;
			}
			
			if(findSplitIntoFibonacci(result, index + 1, i, string.substring(i))) {
				return true;
			}
			
			result.remove(result.size() - 1);
			
		}
		
		return false;
	}
	
	static int maxLen = 0;
	public static int lenLongestFibSubseq(int[] A) {
        List<Integer> result = new ArrayList<>();
        maxLen = 0;
        findLongLen(result, 0, 0, A);
        return maxLen;
    }
	
	public static void findLongLen(List<Integer> result, int index, int start, int[] A) {
		if(result.size() >= 3 && result.size() > maxLen) {
			maxLen = result.size();
		}
		if(start == A.length) {
			
			return;
		}
		
		for(int i = start ; i < A.length ; i++) {
			if(index == 0 || index == 1) {
				result.add(A[i]);
			}else if(A[i] == result.get(index - 1) + result.get(index - 2)) {
				result.add(A[i]);
			}else {
				continue;
			}
			
			findLongLen(result, index + 1, i + 1, A);
			result.remove(result.size() - 1);
		}
	}
	public static int lenLongestFibSubseqs(int[] A) {
		int max = 0 ;
		
		int[][] dp = new int[A.length][A.length];
		for(int i = 0; i < A.length; i++) {
        	Arrays.fill(dp[i], 2);
        }
		
		for(int i = 1 ; i < A.length ; i++) {
			int l = 0;
			int r = i - 1;
			
			while(l < r) {
				int sum = A[l] + A[r];
				
				if(sum == A[i]) {
					dp[l][i] = Math.max(dp[l][i], dp[l][r] + 1);
					max = Math.max(dp[l][i], max);
				}else if(sum < A[i]) {
					l++;
				}else {
					r--;
				}
			}
		}
		return max;
    }
	
	public void setZeroes(int[][] matrix) {
		if(matrix.length == 0) {
			return;
		}
		int[] row = new int[matrix.length];
		int[] col = new int[matrix[0].length];
		for(int i = 0 ; i < matrix.length ; i++) {
			for(int j = 0 ; j < matrix[0].length ; j++) {
				if(matrix[i][j] == 0) {
					row[i] = 1;
					col[j] = 1;
				}
			}
		}
		for(int i = 0 ; i < matrix.length ; i++) {
			for(int j = 0 ; j < matrix[0].length ; j++) {
				if(row[i] == 1 || col[j] == 1) {
					matrix[i][j] = 0;
				}
			}
		}
    }
	
	public int minimumDeleteSum(String s1, String s2) {
        int[][] dp = new int[s2.length() + 1][s1.length() + 1];
        for(int i = 1 ; i <= s1.length() ; i++) {
        	dp[0][i] = (int)s1.charAt(i) + dp[0][i - 1];
        }
        
        for(int i = 1 ; i <= s2.length() ; i++) {
        	dp[i][0] = (int)s2.charAt(i) + dp[i - 1][0];
        }
        
        for(int i = 1 ; i <= s2.length() ; i++) {
        	for(int j = 1 ; j <= s1.length() ; j++) {
        		if(s1.charAt(j - 1) == s2.charAt(i - 1)) {
        			dp[i][j] = Math.min(Math.min(dp[i - 1][j] + (int)s2.charAt(i - 1), dp[i][j - 1] + (int)s1.charAt(j - 1)), dp[i - 1][j - 1]);
        		}else {
        			dp[i][j] = Math.min(dp[i - 1][j] + (int)s2.charAt(i - 1), dp[i][j - 1] + (int)s1.charAt(j - 1));
        		}
        	}
        }
        
        return dp[s2.length()][s1.length()];
    }
	
	
	public int minDistances(String word1, String word2) {
		int[][] dp = new int[word2.length() + 1][word1.length() + 1];
        for(int i = 1 ; i <= word1.length() ; i++) {
        	dp[0][i] = 1 + dp[0][i - 1];
        }
        
        for(int i = 1 ; i <= word2.length() ; i++) {
        	dp[i][0] = 1 + dp[i - 1][0];
        }
        
        for(int i = 1 ; i <= word2.length() ; i++) {
        	for(int j = 1 ; j <= word1.length() ; j++) {
        		if(word1.charAt(j - 1) == word2.charAt(i - 1)) {
        			dp[i][j] = Math.min(Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1]);
        		}else {
        			dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1);
        		}
        	}
        }
        
        return dp[word2.length()][word1.length()];
    }
	
	
	public static int countSubstrings(String s) {
        int count = 0;
        
        boolean[][] dp = new boolean[s.length()][s.length()];
        
        for(int i = 0 ; i < s.length() ; i++) {
        	dp[i][i] = true;
        	count++;
        }
        
        for(int i = 1 ; i < s.length() ; i++) {
        	for(int j = i - 1 ; j >= 0 ; j--) {
        		if(i == j + 1 && s.charAt(i) == s.charAt(j)) {
        			dp[j][i] = true;
        			count++;
        		}else if(dp[j + 1][i - 1] && s.charAt(i) == s.charAt(j)) {
        			dp[j][i] = true;
        			count++;
        		}
        	}
        }
        
        return count;
    }
	
	
	public int longestPalindromeSubseq(String s) {
		String newString = new StringBuilder(s).reverse().toString();
		int[][] dp = new int[s.length() + 1][s.length() + 1];
		for(int i = 1 ; i <= s.length() ; i++) {
			for(int j = 1 ; j <= s.length() ; j++) {
				if(newString.charAt(i - 1) == s.charAt(j - 1)) {
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}else {
					dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
        
        return dp[s.length()][s.length()];
    }
	
	
	public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
	  Integer res = 0;
        Integer n = price.size();
        // System.out.println(n);
        // return n;
        
        for(int i = 0; i < n; i++) {
            res += price.get(i) * needs.get(i);
        }
        
        for(List<Integer> offer:special) {
            // System.out.println(offer);
            boolean flag = true; 
            for(int i = 0; i < n;i++) {
                if(needs.get(i)<offer.get(i)) {
                    flag = false;
                }
                needs.set(i, needs.get(i)-offer.get(i));
            }
            if(flag) {
                res = Math.min(res, shoppingOffers(price, special, needs) + offer.get(n));
            }
            
            for(int i = 0; i < n;i++) {
                needs.set(i, needs.get(i)+offer.get(i));
            }
        }
        return res;
    }
	
	public boolean isMatchs(String s, String p) {
		Pattern pattern = Pattern.compile(p);
		Matcher matcher = pattern.matcher(s);
        return matcher.find();
    }
	
	
	public int minFallingPathSum(int[][] A) {
		int min = Integer.MAX_VALUE;
		for(int i = A.length - 2 ; i >= 0 ; i--) {
			for(int j = 0 ; j < A[i].length ; j++) {
				//三个位置
				int tmp = Integer.MAX_VALUE;
				if(j - 1 >= 0 && tmp > A[i + 1][j - 1]) {
					tmp = A[i + 1][j - 1];
				}
				
				if(j + 1 < A[i].length && tmp > A[i + 1][j + 1]) {
					tmp = A[i + 1][j + 1];
				}
				
				if(tmp > A[i + 1][j]) {
					tmp = A[i + 1][j];
				}
				
				A[i][j] += tmp;
				
			}
		}	
		for(int j = 0 ; j < A[0].length ; j++) {

			if(min > A[0][j]) {
				min = A[0][j];
			}
		}
        return min;
    }
	
	public static boolean canCross(int[] stones) {
		if(stones.length == 1) {
			return true;
		}
		
		if(stones[1] > 1) {
			return false;
		}
		Map<Integer, Integer> stoneIndex = new HashMap<>();
        for(int i = 0 ; i < stones.length ; i++) {
        	stoneIndex.put(stones[i], i);
        }
        
        
        return findWay(stones,stones[stones.length - 1], 1, 1, stoneIndex);
    }
	
	public static boolean findWay(int[] stones,int target, int step, int index, Map<Integer, Integer> stoneIndex) {
		if(stones[index] == target) {
			return true;
		}
		

		if(stoneIndex.get(stones[index] + step + 1) != null && index < stoneIndex.get(stones[index] + step + 1)) {
			if(findWay(stones, target, step + 1, stoneIndex.get(stones[index] + step + 1), stoneIndex)) {
				 return true;	
			};
		}
		if(stoneIndex.get(stones[index] + step - 1) != null && step - 1 > 0 && index < stoneIndex.get(stones[index] + step - 1)) {
			if(findWay(stones,target, step - 1, stoneIndex.get(stones[index] + step - 1), stoneIndex)) {
				 return true;	
			};
		}
		if(stoneIndex.get(stones[index] + step) != null && index < stoneIndex.get(stones[index] + step)) {
			if(findWay(stones, target, step, stoneIndex.get(stones[index] + step), stoneIndex)) {
				 return true;	
			};
		}
		
		return false;
	}

	public List<TreeNode> generateTrees(int n) {
	    if(n == 0)
	        return new LinkedList<TreeNode>();
	    return generateTrees(1,n);
	}
	public List<TreeNode> generateTrees(int start,int end) {
	    List<TreeNode> res = new LinkedList<TreeNode>();
	    if(start > end){
	        res.add(null);
	        return res;
	    }
	    for(int i = start;i <= end;i++){
	        List<TreeNode> subLeftTree = generateTrees(start,i-1);
	        List<TreeNode> subRightTree = generateTrees(i+1,end);
	        for(TreeNode left : subLeftTree){
	            for(TreeNode right : subRightTree){
	                TreeNode node = new TreeNode(i);
	                node.left = left;
	                node.right = right;
	                res.add(node);
	            }
	        }                        
	    }
	    return res;
	}
	
	public int integerBreak(int n) {
		int[] dp = new int[n + 1];
		dp[1] = 1;
		dp[2] = 2;
		dp[3] = 3;
		for(int i = 4 ; i <= n ; i++) {
			for(int j = 2 ; j < n ;j++) {
				dp[i] = Math.max(dp[i], Math.max((dp[i - j]), i - j) * j);
			}
		}
		
		return dp[n];
    }
	
	public static int mincostTickets(int[] days, int[] costs) {
		
        int[] dp = new int[366];
        
        for(int i = 1 ; i < days[days.length - 1] + 1 ; i++) {
        	if(!is_in_days(i, days)) {
        		dp[i] = dp[i - 1];
        	}else {
        		dp[i] = Math.min(dp[i - 1] + costs[0], Math.min((i - 7 > 0 ? dp[i - 7] : 0) + costs[1], (i - 30 > 0 ? dp[i - 30] : 0) + costs[2]));
        	}
        }
        
        return dp[days[days.length - 1]];
    }
	
	public static boolean is_in_days(int n, int[] days) {
		for(int i = 0 ; i < days.length ; i++) {
			if(n == days[i]) {
				return true;
			}
		}
		return false;
	}
	
	
	public static int change(int amount, int[] coins) {
	        int[][] dp = new int[coins.length + 1][amount+1];
	        
	        Arrays.sort(coins);
	        for(int i = 0 ; i <= coins.length; i++) {
	        	dp[i][0] = 1;
	        }
	        
	        for(int i = 1 ; i <= coins.length ; i++) {
	        	for(int j = 1 ; j <= amount ; j++) {
	        		for(int k = j ; k >= 0 ; k -= coins[i - 1]) {
	        			dp[i][j] += dp[i - 1][k];
	        		}
	        	}
	        }
	        
	        return dp[coins.length][amount];
	   }
	public int[] maxSlidingWindow(int[] nums, int k) {
		Queue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {

			@Override
			public int compare(Integer o1, Integer o2) {
				if(o1.intValue() > o2.intValue()) {
					return -1;
				}else if(o1.intValue() < o2.intValue()) {
					return 1;
				}
				return 0;
			}
		});
		
		for(int i = 0 ; i < k - 1 ; i++) {
			queue.add(nums[i]);
		}
		int[] result = new int[nums.length - k + 1];
		int index = 0;
		for(int i = k - 1 ; i < nums.length ; i++) {
			queue.add(nums[i]);
			result[index] = queue.peek();
			queue.remove(nums[index++]);
		}
		
		return result;
    }
	
	public int deleteAndEarn(int[] nums) {
		int max = 0;
		for(int i = 0 ; i < nums.length ; i++) {
			if(max < nums[i]) {
				max = nums[i];
			}
		}
		int[] values = new int[max + 1];
		for(int i = 0 ; i < nums.length ; i++) {
			values[nums[i]] += nums[i];
		}
		
		if(values.length == 0) {
			return 0;
		}
		
		if(values.length == 2) {
			return values[1];
		}
		int[] dp = new int[max + 1];
		dp[1] = values[1];
		for(int i = 2 ; i <= max ; i++) {
			dp[i] = Math.max(dp[i - 1], dp[i] + values[i]);
		}
		return Math.max(dp[max - 1], dp[max] + values[max]);	
    }
	
	
	public int rob(TreeNode root) {
       return Math.max(rob(root, true), rob(root, false));
    }
	
	public int rob(TreeNode root, boolean flag) {
		
		if(root == null) {
			return 0;
		}
		if(flag) {
			return rob(root.left, false) + rob(root.right, false);
		}else {
			return Math.max(rob(root.left, false) + rob(root.right, false), rob(root.left, true) + rob(root.right, true) + root.val);
		}
	}
	
	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		if(root == null) {
			return null;
		}
		Stack<TreeNode> stack1 = new Stack<>();
		Stack<TreeNode> stack2 = new Stack<>();
		
		findNode(stack1, root, p);
		findNode(stack2, root, q);
		
		while(stack1.size() > stack2.size()) {
			stack1.pop();
		}
		
		while(stack2.size() > stack1.size()) {
			stack2.pop();
		}
		
		while(stack1.peek() != stack2.peek()) {
			stack1.pop();
			stack2.pop();
		}
		
		return stack1.pop();
    }
	
	public boolean findNode(Stack<TreeNode> stack, TreeNode root, TreeNode target) {
		if(root == target) {
			stack.push(root);
			return true;
		}
		if(root == null) {
			return false;
		}
		
		boolean flag = false;
		stack.push(root);
		flag = findNode(stack, root.left, target);
		if(!flag) {
			flag = findNode(stack, root.right, target);
		}
		if(!flag) {
			stack.pop();
		}
		return flag;
	}
	
	public int maxProfits(int[] prices) {
		int[] sell = new int[prices.length];
		int[] buy = new int[prices.length];
		int[] cold = new int[prices.length];
		
		buy[0] = -prices[0];
		for(int i = 1; i < prices.length; i++) {
			sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
			buy[i] = Math.max(cold[i - 1] - prices[i], buy[i - 1]);
			cold[i] = Math.max(Math.max(cold[i - 1], sell[i - 1]), buy[i - 1]);
		}
		return sell[prices.length - 1];
    }
	
	public List<List<Integer>> levelOrderBottom(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        TreeNode last = root;
        TreeNode next = null;
        TreeNode cur = null;
        List<Integer> temp = new ArrayList<>();
        Stack<List<Integer>> stack = new Stack<>();
        queue.add(root);
        while(!queue.isEmpty()) {
        	cur = queue.poll();
        	temp.add(cur.val);
        	if(cur.right != null) {
        		next = cur.right;
        	}else if(cur.left != null){
        		next = cur.left;
        	}
        	
        	if(cur == last) {
        		last = next;
        		stack.push(temp);
        		temp = new ArrayList<>();
        	}
        }
        
        List<List<Integer>> result = new ArrayList<>();
        
	    while(!stack.isEmpty()) {
	    	result.add(stack.pop());
	    }
        return result;
    }
	
	public int findLongestChain(int[][] pairs) {
        for(int i = 0 ; i < pairs.length ; i++) {
        	for(int j = 0 ; j < pairs.length - i - 1 ; j++) {
        		if(pairs[j][1] > pairs[j + 1][1]) {
        			int[] temp = pairs[j];
        			pairs[j] = pairs[j + 1];
        			pairs[j + 1] = temp;
        		}else if(pairs[j][1] == pairs[j + 1][1] && pairs[j][0] > pairs[j + 1][0]) {
        			int[] temp = pairs[j];
        			pairs[j] = pairs[j + 1];
        			pairs[j + 1] = temp;
        		}
        	}
        }
        int end = pairs[0][1];
        int count = 1;
        for(int i = 1 ; i < pairs.length ; i++) {
        	if(pairs[i][0] > end) {
        		count++;
        		end = pairs[i][1];
        	}
        }
        return count;
    }
	public List<List<Integer>> findSubsequences(int[] nums) {
		Set<List<Integer>> result = new HashSet<>();
		dfsfindSubsequences(result, new ArrayList<>(), nums, 0);
		return new ArrayList<>(result);
    }
	
	public void dfsfindSubsequences(Set<List<Integer>> result, List<Integer> temp, int[] nums, int index) {
		if(temp.size() >= 2 && !result.contains(temp)) {
			result.add(temp);
		}
		
		if(index == nums.length) {
			return ;
		}
		List<Integer> nextTmp = new ArrayList<>(temp);
		for(int i = index ; i < nums.length ; i++) {
			
			if(nextTmp.size() == 0) {
				nextTmp.add(nums[i]);
			}else if(nextTmp.get(nextTmp.size() - 1) <= nums[i]) {
				nextTmp.add(nums[i]);
			}
			dfsfindSubsequences(result, nextTmp, nums, i + 1);
			nextTmp = new ArrayList<>(nextTmp);
			nextTmp.remove(nextTmp.size() - 1);
		}
	}
	
	private int[] row = {-1,1,0,0};
    private int[] col = {0,0,-1,1};
	public int longestIncreasingPath(int[][] matrix) {
		if(matrix.length == 0) {
			return 0;
		}
		int[][] dp = new int[matrix.length][matrix[0].length];
		int matrixMax = 0;
		boolean[][] used = new boolean[matrix.length][matrix[0].length];
		for(int i = 0 ; i < matrix.length ; i++) {
			for(int j = 0 ; j < matrix[0].length ; j++) {
				matrixMax = Math.max(findLongWay(matrix, used, i, j,dp), matrixMax);
			}
		}
        return matrixMax;
    }
	
	public int findLongWay(int[][] matrix, boolean[][] used, int x, int y,int[][] dp) {
		if(used[x][y]){
			return dp[x][y];
		}
		dp[x][y] = 1;
		
		if(x - 1 >= 0 && matrix[x - 1][y] > matrix[x][y]) {
			dp[x][y] = Math.max(findLongWay(matrix, used, x - 1, y, dp) + 1, dp[x][y]);
		}
		
		if(x + 1 < matrix.length && matrix[x + 1][y] > matrix[x][y]) {
			dp[x][y] = Math.max(findLongWay(matrix, used, x + 1, y, dp) + 1, dp[x][y]);
		}
		
		if(y + 1 < matrix[0].length && matrix[x][y + 1] > matrix[x][y]) {
			
			dp[x][y] = Math.max(findLongWay(matrix, used, x, y + 1, dp) + 1, dp[x][y]);
		}
		
		if(y - 1 >= 0 && matrix[x][y - 1] > matrix[x][y]) {
			dp[x][y] = Math.max(findLongWay(matrix, used, x, y - 1, dp) + 1, dp[x][y]);
		}
		
		used[x][y] = true;
		return dp[x][y];
	}
	
	public String minWindow(String s, String t) {
		if(t.length() == 0) {
			return "";
		}
		int[] array = new int[26];
		for(int i = 0 ; i < t.length() ; i++) {
			array[t.charAt(i) - 'a']++;
		}
		String string = "";
		int[] record = new int[s.length()];
		int index = 0;
		for(int i = 0 ; i < s.length() ; i++) {
			if(array[s.charAt(i) - 'a'] > 0) {
				record[index++] = i;
				string += s.charAt(i);
			}
		}
		
		int l = 0, r = 0, count = 0;
		int min = Integer.MAX_VALUE,start = 0, end = 0;
		int[] newArray = Arrays.copyOf(array, array.length);
		for(; l < index && r < index ;) {
			if(newArray[string.charAt(r)] > 0) {
				newArray[string.charAt(r)]--;
				count++;
				
			}
			
			if(count == t.length()) {
				if(min > record[r] - record[l]) {
					min = record[r] - record[l];
					start = record[l];
					end = record[r];
				}
				l++;
				newArray = Arrays.copyOf(array, array.length);
				r = l - 1;
			}
			r++;	
		}
		return start == end ? "" : s.substring(start, end + 1);
     }
	
	public int maxProduct(int[] nums) {
		int count = 0;
		for(int i = 0 ; i < nums.length ; i++) {
			if(nums[i] < 0) {
				count++;
			}
		}
		
		int max = Integer.MIN_VALUE, temp = count, value = 1;;
		for(int i = 0 ; i < nums.length ; i++) {
			temp = count;
			value = 1;
			for(int j = i ; j < nums.length ; j++) {
				if(value == 0 || (value < 0 && temp == 0)) {
					
					break;
				}
				value *= nums[j];
				if(value > max) {
					max = value;
				}
				
				if(nums[j] < 0){
                    temp--;
                }
			}
		}
		return max;
    }
	
	public double findMedianSortedArrays(int[] nums1, int[] nums2) {
		int i = 0, j = 0;
		int sum = nums1.length + nums2.length;
		boolean ave = false;
		if(sum % 2 == 0) {
			ave = true;
		}
		
		sum /= 2;
		
		while(i < nums1.length && j < nums2.length && i + j + 1 < sum) {
			if(nums1[i] > nums2[j]) {
				i++;
			}else {
				j++;
			}
		}
		
		if(i == nums1.length) {
			while(i + j + 1 < sum) {
				j++;
			}
			
			if(ave) {
				return (nums2[j] + nums2[j + 1]) / 2.0;
			}else {
				return (double)nums2[j];
			}
		}
		
		
		if(j == nums2.length) {
			while(i + j + 1 < sum) {
				i++;
			}
			
			if(ave) {
				return (nums1[i] + nums2[i + 1]) / 2.0;
			}else {
				return (double)nums1[i];
			}
		}
		
		if(ave) {
			return (nums1[i] + nums2[j]) / 2.0;
		}else {
			return (double)(nums1[i] > nums2[j] ? nums1[i] : nums2[j]);
		}
			
    }
	
	public static List<Integer> findSubstringss(String s, String[] words) {
		if(words.length == 0) {
			return new ArrayList<>();
		}
        int wordLen = words[0].length();
        Map<String, Integer> map = new HashMap<>();
        
        for(int i = 0 ; i < words.length ; i++) {
        	if(map.get(words[i]) == null) {
        		map.put(words[i], 1);
        	}else {
        		map.put(words[i], map.get(words[i]) + 1);
        	}
        }
        int count = 0;
        List<Integer> result = new ArrayList<>();
        for(int i = 0 ; i <= s.length() - wordLen * words.length ; i++) {
        	count = 0;
        	Map<String, Integer> tmpmap = new HashMap<>();
        	for(int j = i ; j < i + wordLen * words.length ; j+= wordLen) {
        		String tmp = s.substring(j, j + wordLen);
        		if(tmpmap.get(tmp) == null) {
        			tmpmap.put(tmp, 1);
        		}else {
        			tmpmap.put(tmp, tmpmap.get(tmp) + 1);
        		}
        		
        		if(map.get(tmp) == null || tmpmap.get(tmp) > map.get(tmp)) {
        			break;
        		}else {
        			count++;
        		}
        		
        		if(count == words.length) {
        			result.add(i);
        		}
        	}
        }
        return result;
    }
	public static int[] maxSlidingWindows(int[] nums, int k) {
		if(nums.length == 0 || nums.length < k) {
			return new int[0];
		}
        int[] result = new int[nums.length - k];
        int index = 0;
        LinkedList<Integer> list = new LinkedList<>();
        for(int i = 0 ; i < nums.length ; i++) {
        	if(list.size() == 0 || list.get(list.size() - 1) >= nums[i]) {
        		list.add(nums[i]);
        	}else {
        		int count = 0;
        		while(list.size() > 0 && list.get(list.size() - 1) < nums[i]) {
        			list.remove(list.size() - 1);
        			count++;
        		}
        		while(count >= 0) {
        			list.add(nums[i]);
        			count--;
        		}
        	}
        	if(list.size() == k) {
        		result[index++] = list.removeFirst();
        	}
        }
        
        return result;
    }
	
    public static void main(String[] args) throws Exception{
    	 Object object = new Object();
    	new Thread(new Runnable() {
			
			@Override
			public void run() {
				System.out.println(object);
				
			}
		}).start();
	}
}

class A extends B{
	public int a = 100;
	public A() {
		System.out.println(a);
		a = 200;
		System.out.println(a);
		
	}
}

class B{
	public B() {
		System.out.println(((A)this).a);
		((A)this).a = 50;
	}
}

class Base implements Serializable{
	public String string = "base";
	List<Sub> subs = new ArrayList<>();
	public String getString() {
		return string;
	}
	public void setString(String string) {
		this.string = string;
	}
	public List<Sub> getSubs() {
		return subs;
	}
	public void setSubs(List<Sub> subs) {
		this.subs = subs;
	}
	
}

class Sub implements Serializable{
	public String string = "Sub";
	List<Base> bases = new ArrayList<>();
	public String getString() {
		return string;
	}
	public void setString(String string) {
		this.string = string;
	}
	public List<Base> getBases() {
		return bases;
	}
	public void setBases(List<Base> bases) {
		this.bases = bases;
	}

}

class Point {
   int x;
    int y;
    Point() { x = 0; y = 0; }
    Point(int a, int b) { x = a; y = b; }
 }

class LRUCache extends LinkedHashMap<Integer, Integer>{
	private int capacity;
    public LRUCache(int capacity) {
        super(capacity, 0.75f, true);
        this.capacity = capacity;
    }
    
    @Override
	protected boolean removeEldestEntry(Entry<Integer, Integer> eldest) {
		if(this.size() > this.capacity) {
			return true;
		}
		
		return false;
	}

	public int get(int key) {
        Integer value = super.get(key);
        return value == null ? 0 : value;
    }
    
    public void put(int key, int value) {
        super.put(key, value);
    }
}

class Trie {
	TrieNode root;
    /** Initialize your data structure here. */
    public Trie() {
    	root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode node = root;
        for(int i = 0 ; i < word.length() ; i++) {
        	char ch = word.charAt(i);
        	if(node.childs[ch - 'a'] == null) {
        		node.childs[ch - 'a'] = new TrieNode();
        	}
        	
        	node = node.childs[ch - 'a'];
        }
        node.isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
    	TrieNode node = root;
    	for(int i = 0 ; i < word.length() ; i++) {
    		char ch = word.charAt(i);
        	if(node.childs[ch - 'a'] == null) {
        		return false;
        	}
        	node = node.childs[ch - 'a'];
    	}
        return node.isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
    	TrieNode node = root;
    	for(int i = 0 ; i < prefix.length() ; i++) {
    		char ch = prefix.charAt(i);
        	if(node.childs[ch - 'a'] == null) {
        		return false;
        	}
        	node = node.childs[ch - 'a'];
    	}
    	return true;
    }
    
    class TrieNode{
    	TrieNode[] childs;
    	boolean isEnd;
    	public TrieNode() {
    		childs = new TrieNode[26];
    		isEnd = false;
    	}
    }
}

class Person implements Cloneable{
	StringBuffer name;
	
	public Person(String name) {
		this.name = new StringBuffer(name);
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return super.clone();
	}
	
}

class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; }
 }

class Interval {
	int start;
	int end;
	Interval() { start = 0; end = 0; }
	Interval(int s, int e) { start = s; end = e; }
}


class RandomListNode {
	 int label;
	 RandomListNode next, random;
	 RandomListNode(int x) { this.label = x; }
};



class MyStack {
	private Queue queue;
    /** Initialize your data structure here. */
    public MyStack() {
        queue= new ArrayDeque<>();
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
    	Queue temp = new ArrayDeque<>();
    	while(!queue.isEmpty()) {
    		temp.add(queue.poll());
    	}
    	
    	queue.add(x);
    	
    	while(!temp.isEmpty()) {
    		queue.add(temp.poll());
    	}
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
      return  (int) queue.poll();
    }
    
    /** Get the top element. */
    public int top() {
    	return (int) queue.peek();
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
    	return queue.isEmpty();
        
    }
}


class MinStack {
	private Stack<Integer> stack;
	private Stack<Integer> minStack;
    /** initialize your data structure here. */
    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }
    
    public void push(int x) {
        stack.push(x);
        if(stack.isEmpty()) {
        	minStack.add(x);
        }else {
        	if(x < minStack.peek()) {
            	minStack.add(x);
            }else {
            	minStack.add(minStack.peek());
            }
        }
    }
    
    public void pop() {
        minStack.pop();
        stack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int getMin() {
        return minStack.peek();
    }
}


class TreeNode {
	 int val;
	 TreeNode left;
	 TreeNode right;
	 TreeNode(int x) { val = x; }
}

class Car implements Serializable {

	private static final long serialVersionUID = -5713945027627603702L;

	private String brand; // 品牌
	private int maxSpeed; // 最高时速

	public Car(String brand, int maxSpeed) {

		this.brand = brand;

		this.maxSpeed = maxSpeed;

	}

	public String getBrand() {
		return brand;

	}

	public void setBrand(String brand) {

		this.brand = brand;

	}

	public int getMaxSpeed() {

		return maxSpeed;

	}

	public void setMaxSpeed(int maxSpeed) {

		this.maxSpeed = maxSpeed;

	}

	@Override

	public String toString() {

		return "Car [brand=" + brand + ", maxSpeed=" + maxSpeed + "]";

	}
}

