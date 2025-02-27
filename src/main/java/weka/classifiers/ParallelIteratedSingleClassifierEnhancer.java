/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ParallelIteratedSingleClassifierEnhancer.java
 *    Copyright (C) 2009-2014 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

/**
 * Abstract utility class for handling settings common to meta classifiers that
 * build an ensemble in parallel from a single base learner.
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @author Bernhard Pfahringer (bernhard@waikato.ac.nz)
 * @version $Revision: 11909 $
 */
public abstract class ParallelIteratedSingleClassifierEnhancer extends
  IteratedSingleClassifierEnhancer {

  /** For serialization */
  private static final long serialVersionUID = -5026378741833046436L;

  /** The number of threads to have executing at any one time */
  protected int m_numExecutionSlots = 1;

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    Vector<Option> newVector = new Vector<Option>(2);

    newVector.addElement(new Option("\tNumber of execution slots.\n"
      + "\t(default 1 - i.e. no parallelism)\n"
      + "\t(use 0 to auto-detect number of cores)", "num-slots", 1,
      "-num-slots <num>"));

    newVector.addAll(Collections.list(super.listOptions()));

    return newVector.elements();
  }

  /**
   * Parses a given list of options. Valid options are:
   * <p>
   *
   * -num-slots num <br>
   * Set the number of execution slots to use (default 1 - i.e. no parallelism).
   * <p>
   *
   * Options after -- are passed to the designated classifier.
   * <p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {

    String iterations = Utils.getOption("num-slots", options);
    if (iterations.length() != 0) {
      setNumExecutionSlots(Integer.parseInt(iterations));
    } else {
      setNumExecutionSlots(1);
    }

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {

    String[] superOptions = super.getOptions();
    String[] options = new String[superOptions.length + 2];

    int current = 0;
    options[current++] = "-num-slots";
    options[current++] = "" + getNumExecutionSlots();

    System.arraycopy(superOptions, 0, options, current, superOptions.length);

    return options;
  }

  /**
   * Set the number of execution slots (threads) to use for building the members
   * of the ensemble.
   *
   * @param numSlots the number of slots to use.
   */
  public void setNumExecutionSlots(int numSlots) {
    m_numExecutionSlots = numSlots;
  }

  /**
   * Get the number of execution slots (threads) to use for building the members
   * of the ensemble.
   *
   * @return the number of slots to use
   */
  public int getNumExecutionSlots() {
    return m_numExecutionSlots;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String numExecutionSlotsTipText() {
    return "The number of execution slots (threads) to use for "
      + "constructing the ensemble.";
  }

  /**
   * Stump method for building the classifiers
   *
   * @param data the training data to be used for generating the ensemble
   * @exception Exception if the classifier could not be built successfully
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {
    super.buildClassifier(data);

    //if (m_numExecutionSlots < 0) {
      //throw new Exception("Number of execution slots needs to be >= 0!");
    //}
  }

  /**
   * Start the pool of execution threads
   */

  /**
   * Does the actual construction of the ensemble
   *
   * @throws Exception if something goes wrong during the training process
   */
  protected void buildClassifiers() throws Exception {

    if ((m_numExecutionSlots != 1) && (m_numExecutionSlots != -1)) {

      int numCores =
        (m_numExecutionSlots == 0) ? Runtime.getRuntime().availableProcessors()
          : (m_numExecutionSlots < 0) ? -m_numExecutionSlots : m_numExecutionSlots;
      
      ExecutorService executorPool = Executors.newFixedThreadPool(numCores);
      
      final CountDownLatch doneSignal =
		        new CountDownLatch(m_Classifiers.length);

      final AtomicInteger numFailed = new AtomicInteger();
      
      if (m_numExecutionSlots >= 0) {
    	  
    	  
    	  for (int i = 0; i < m_Classifiers.length; i++) {

    	        final Classifier currentClassifier = m_Classifiers[i];
    	        // MultiClassClassifier may produce occasional NULL classifiers ...
    	        if (currentClassifier == null)
    	          continue;
    	        final int iteration = i;

    	        if (m_Debug) {
    	          System.out.print("Training classifier (" + (i + 1) + ")");
    	        }
    	        Runnable newTask = new Runnable() {
    	          @Override
    	          public void run() {
    	            try {
    	              long start, end;
    	              start = System.currentTimeMillis();
    	              currentClassifier.buildClassifier(getTrainingSet(iteration));
    	              end = System.currentTimeMillis();
    	              if (m_Debug) {
    	    	          System.out.print("dynamic classifier (" + (iteration + 1) + "), time: " + (end-start) + " \n");
    	    	      }
    	            } catch (Throwable ex) {
    	              ex.printStackTrace();
    	              numFailed.incrementAndGet();
    	              if (m_Debug) {
    	                System.err.println("Iteration " + iteration + " failed!");
    	              }
    	            } finally {
    	              doneSignal.countDown();
    	            }
    	          }
    	        };
    	        // launch this task
    	        executorPool.submit(newTask);
    	      }
    	  doneSignal.await();
      } else {
    	  
    	  for (int threadIndex = 0; threadIndex < numCores; threadIndex++) {
    		    final int assignedThread = threadIndex;
    		    executorPool.submit(() -> {
    		        for (int i = assignedThread; i < m_Classifiers.length; i += numCores) {
    		            final Classifier currentClassifier = m_Classifiers[i];
    		            if (currentClassifier == null) continue;

    		            try {
    		                long start = System.currentTimeMillis();
    		                currentClassifier.buildClassifier(getTrainingSet(i));
    		                long end = System.currentTimeMillis();
    		                if (m_Debug) {
    		                    System.out.println("Static classifier (" + (i + 1) + "), time: " + (end - start));
    		                }
    		            } catch (Throwable ex) {
    		                ex.printStackTrace();
    		                numFailed.incrementAndGet();
    		                if (m_Debug) {
    		                    System.err.println("Iteration " + i + " failed!");
    		                }
    		            }
    		            doneSignal.countDown();
    		        }
    		    });
    		}
    		doneSignal.await();
    		
      }
    	  /**final int[] core_div = new int[numCores];
    	  for (int k = 0; k < numCores; k++) {
    		  core_div[k] = m_Classifiers.length/numCores;
    	  }
    	  if (m_Classifiers.length % numCores != 0) for (int m = 0; m < (m_Classifiers.length % numCores); m++) core_div[m] += 1;
    	  
    	  int loop_core = (m_Classifiers.length < numCores) ? (m_Classifiers.length % numCores) : numCores;
    	  
    	  final int[] core_carry = new int[loop_core];
    	  core_carry[0] = 0;
    	  for (int k=1; k<loop_core; k++) core_carry[k] = core_carry[k-1] + core_div[k-1];
    	  
    	  List<Thread> baggingThreads = new ArrayList<>();
    	  
    	  for (int i_core=0; i_core < loop_core; i_core++) {
    		  
    		  final int current_core = i_core;
    		  
    		  Thread bagThread = new Thread(new Runnable() {
    			  @Override
    			  public void run() {
    				  try {
    					  for (int i = 0; i < (core_div[current_core]); i++) {
    						  int index = core_carry[current_core]+i;
    						  Classifier currentClassifier = m_Classifiers[index];
    			    	        // MultiClassClassifier may produce occasional NULL classifiers ...
    			    	        if (currentClassifier == null)
    			    	          continue;
    			    	        //int iteration = index;

    			    	        //if (m_Debug) {
    			    	          //System.out.print("Training classifier (" + (iteration + 1) + ")");
    			    	        //}
    			    	        long start, end;
    		    	              start = System.currentTimeMillis();
    		    	              currentClassifier.buildClassifier(getTrainingSet(index));
    		    	              end = System.currentTimeMillis();
    		    	              if (m_Debug) {
    		    	    	          System.out.print("static classifier (" + (index + 1) + "), time: " + (end-start) + " by thread: " + Thread.currentThread().getName() + " \n");
    		    	    	      }
    			    	       // if (m_Debug) {
      			    	         // System.out.print("Classifier (" + (iteration + 1) + ") trained! \n");
      			    	        //}

        				  }
    				  } catch (Throwable e) {
    					  e.printStackTrace();
    				  } 
    			  }
    		  });
    		  bagThread.start();
    		  baggingThreads.add(bagThread);
    	  }
    	  
    	  for (Thread bagThread: baggingThreads) {
    		  try {
    			  bagThread.join();
    		  } catch (InterruptedException e) {
    			  e.printStackTrace();
    		  }
    	  }
      }**/
      
      
      executorPool.shutdown();
     
      if (m_Debug && numFailed.intValue() > 0) {
        System.err
          .println("Problem building classifiers - some iterations failed.");
      }

    } else {
      // simple single-threaded execution
      for (int i = 0; i < m_Classifiers.length; i++) {
        m_Classifiers[i].buildClassifier(getTrainingSet(i));
      }
    }
  }

  /**
   * Gets a training set for a particular iteration. Implementations need to be
   * careful with thread safety and should probably be synchronized to be on the
   * safe side.
   *
   * @param iteration the number of the iteration for the requested training set
   * @return the training set for the supplied iteration number
   * @throws Exception if something goes wrong.
   */
  protected abstract Instances getTrainingSet(int iteration) throws Exception;
}
