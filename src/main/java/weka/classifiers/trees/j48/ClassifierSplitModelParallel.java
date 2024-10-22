package weka.classifiers.trees.j48;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.Utils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class ClassifierSplitModelParallel extends ClassifierSplitModel implements Cloneable, Serializable, RevisionHandler {

	private static final long serialVersionUID = -1986057555442699039L;
	
	/**
	   * Splits the given set of instances into subsets.
	   *
	   * @exception Exception if something goes wrong
	   */
	  public Instances [] splitParallel(Instances data, int numCore) throws Exception { 
		  
		ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
			
		final CountDownLatch doneSignal = new CountDownLatch(data.size());
			
		final AtomicInteger numFailed = new AtomicInteger();

	    // Find size and constitution of subsets
	    int[] subsetSize = new int[m_numSubsets];
	    for (Instance instance : data) {
	      Runnable subsetSizeTask = new Runnable() {
	    	  public void run() {
	    		  try {
	    			  int subset = whichSubset(instance);
	    		      if (subset > -1) {
	    		        subsetSize[subset]++;
	    		      } else {
	    		        double[] weights = weights(instance);
	    		        for (int j = 0; j < m_numSubsets; j++) {
	    		          if (Utils.gr(weights[j], 0)) {
	    		            subsetSize[j]++;
	    		          }
	    		        }
	    		      }
	    		  } catch(Throwable e) {
	    			e.printStackTrace();
					numFailed.incrementAndGet();
					System.out.println("SplitParallel failed!");
	    		  } finally {
	    			  doneSignal.countDown();
	    		  }
	    	  }
	      };
	      executorPool.submit(subsetSizeTask);
	    }
	    doneSignal.await();
	    
	    // Create subsets
	    Instances [] instances = new Instances [m_numSubsets];
	    for (int j = 0; j < m_numSubsets; j++) {
	      instances[j] = new Instances(data, subsetSize[j]);
	    }
	    for (Instance instance : data) {
	      int subset = whichSubset(instance);
	      if (subset > -1) {
		instances[subset].add(instance);
	      } else {
	        double[] weights = weights(instance);
	        for (int j = 0; j < m_numSubsets; j++) {
		  if (Utils.gr(weights[j], 0)) {
		    instances[j].add(instance);
		    instances[j].lastInstance().
		      setWeight(weights[j] * instance.weight());
		  }
		}
	      }
	    }
	    
	    return instances;
	  }

}
