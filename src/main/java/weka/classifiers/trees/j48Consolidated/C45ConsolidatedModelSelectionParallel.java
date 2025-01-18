package weka.classifiers.trees.j48Consolidated;

import weka.classifiers.trees.j48.*;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.matrix.DoubleVector;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class C45ConsolidatedModelSelectionParallel extends C45ConsolidatedModelSelection {

	
	private static final long serialVersionUID = 5464380024740285421L;
	
	protected ModelSelection m_toSelectModelToConsolidateParallel;
	
	public C45ConsolidatedModelSelectionParallel(int minNoObj, Instances allData,
			boolean useMDLcorrection, boolean doNotMakeSplitPointActualValue) {
		super(minNoObj, allData, useMDLcorrection, doNotMakeSplitPointActualValue);
		
		m_toSelectModelToConsolidateParallel = new C45ConsolidatedModelSelection(minNoObj, allData,
				useMDLcorrection, doNotMakeSplitPointActualValue);
	}
	
	/**
	 * Getter of m_toSelectModelToConsolidateParallel
	 * @return the m_toSelectModelToConsolidateParallel
	 */
	public ModelSelection getModelToConsolidateParallel() {
		return m_toSelectModelToConsolidateParallel;
	}
	
	
	/**
	 * Selects Consolidated-type split based on C4.5 for the given dataset concurrently.
	 * 
	 * @param data the data to train the classifier with
	 * @param samplesVector the vector of samples
	 * @param numCore the number of threads going to be used to select the Consolidated-type split
	 * @return the consolidated model to be used to split
	 * @throws Exception  if something goes wrong
	 */
	public ClassifierSplitModel selectModelParallel(Instances data, Instances[] samplesVector, int numCore) throws Exception{

		/** Number of Samples. */
		int numberSamples = samplesVector.length;
		/** Vector storing the chosen attribute to split in each sample */
		int[] attIndexVector = new int[numberSamples];
		/** Vector storing the split point to use to split, if numerical, in each sample */
		double[] splitPointVector = new double[numberSamples];
		
		ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
		
		final CountDownLatch doneSignal = new CountDownLatch(numberSamples);
		
		final AtomicInteger numFailed = new AtomicInteger();

		// Select C4.5-type split for each sample and
		//  save the chosen attribute (and the split point if numerical) to split 
		for (int iSample = 0; iSample < numberSamples; iSample++) {
			
			final int currentSample = iSample;
			
			Runnable selectModelTask = new Runnable() {
				public void run() {
					try {
						ClassifierSplitModel localModel = m_toSelectModelToConsolidateParallel.selectModel(samplesVector[currentSample]);
						if(localModel.numSubsets() > 1){
							attIndexVector[currentSample] = ((C45Split) localModel).attIndex();
							splitPointVector[currentSample] = ((C45Split) localModel).splitPoint();
						}else{
							attIndexVector[currentSample] = -1;
							splitPointVector[currentSample] = -1;
						}
					} catch (Throwable e) {
						e.printStackTrace();
						numFailed.incrementAndGet();
						System.err.println("Iteration " + currentSample + " failed!");
					} finally {
						doneSignal.countDown();
					}
				}
			};
			executorPool.submit(selectModelTask);
		}
		
		doneSignal.await();
		executorPool.shutdownNow();
		
		// Get the most voted attribute (index)
		int votesCountByAtt[] = new int[data.numAttributes()];
		int numberVotes = 0;
		for (int iAtt = 0; iAtt < data.numAttributes(); iAtt++)
			votesCountByAtt[iAtt] = 0;
		for (int iSample = 0; iSample < numberSamples; iSample++)
			if(attIndexVector[iSample]!=-1){
				votesCountByAtt[attIndexVector[iSample]]++;
				numberVotes++;
			}
		int mostVotedAtt = Utils.maxIndex(votesCountByAtt);

		Distribution checkDistribution = new DistributionConsolidated(samplesVector);
		NoSplit noSplitModel = new NoSplit(checkDistribution);
		// if all nodes are leafs,
		if(numberVotes==0)
			//  return a consolidated leaf
			return noSplitModel;
		
		// Consolidate the split point (if numerical)
		double splitPointConsolidated = consolidateSplitPoint(mostVotedAtt, attIndexVector, splitPointVector, data);
		// Creates the consolidated model
		C45ConsolidatedSplit consolidatedModel =
				new C45ConsolidatedSplit(mostVotedAtt, m_minNoObj, checkDistribution.total(), 
						m_useMDLcorrection, data, samplesVector, splitPointConsolidated);

//		// Set the split point analogue to C45 if attribute numeric.
//		// // It is not necessary for the consolidation process because the median value 
//		// //  is already one of the proposed split points.
//		consolidatedModel.setSplitPoint(data);
		
		if(!consolidatedModel.checkModel())
			return noSplitModel;
		return consolidatedModel;
	}
	
	
	/**
	 * Selects Consolidated-type split based on C4.5 for the given dataset concurrently (static Multihreading).
	 * 
	 * @param data the data to train the classifier with
	 * @param samplesVector the vector of samples
	 * @param numCore the number of threads going to be used to select the Consolidated-type split
	 * @return the consolidated model to be used to split
	 * @throws Exception  if something goes wrong
	 */
	public ClassifierSplitModel selectModelParallel_static(Instances data, Instances[] samplesVector, int numCore) throws Exception{

		/** Number of Samples. */
		int numberSamples = samplesVector.length;
		/** Vector storing the chosen attribute to split in each sample */
		int[] attIndexVector = new int[numberSamples];
		/** Vector storing the split point to use to split, if numerical, in each sample */
		double[] splitPointVector = new double[numberSamples];
		
		final int[] core_div_samples = new int[numCore];
  	  	for (int k = 0; k < numCore; k++) {
  	  		core_div_samples[k] = numberSamples/numCore;
  	  	}
  	  	if (numberSamples % numCore != 0) for (int m = 0; m < (numberSamples % numCore); m++) core_div_samples[m] += 1;
  	  
	  	int loop_core_samples = (numberSamples < numCore) ? (numberSamples % numCore) : numCore;
	  
	  	final int[] core_carry_samples = new int[loop_core_samples];
	  	core_carry_samples[0] = 0;
	  	for (int k=1; k<loop_core_samples; k++) core_carry_samples[k] = core_carry_samples[k-1] + core_div_samples[k-1];
	  	
	  	List<Thread> numberSamplesThreads = new ArrayList<>();
	  	
	  	for (int i_core=0; i_core<loop_core_samples; i_core++) {
	  		
	  		final int current_core = i_core;
	  		
	  		Thread selectModelThread = new Thread(new Runnable() {
	  			@Override
	  			public void run() {
	  				try {
	  					for (int i = 0; i < (core_div_samples[current_core]); i++) {
	  						int selectModelIndex = core_carry_samples[current_core]+i;
	  						ClassifierSplitModel localModel = m_toSelectModelToConsolidateParallel.selectModel(samplesVector[selectModelIndex]);
							if(localModel.numSubsets() > 1){
								attIndexVector[selectModelIndex] = ((C45Split) localModel).attIndex();
								splitPointVector[selectModelIndex] = ((C45Split) localModel).splitPoint();
							}else{
								attIndexVector[selectModelIndex] = -1;
								splitPointVector[selectModelIndex] = -1;
							}
	  					}
	  				} catch (Throwable e) {
	  					e.printStackTrace();
	  				}
	  			}
	  		});
	  		selectModelThread.start();
	  		numberSamplesThreads.add(selectModelThread);
	  	}
	  	
	  	for (Thread selectModelThread: numberSamplesThreads) {
	  		  try {
	  			  selectModelThread.join();
			  } catch (InterruptedException e) {
				  e.printStackTrace();
			  }
		  }
		
		// Get the most voted attribute (index)
		int votesCountByAtt[] = new int[data.numAttributes()];
		int numberVotes = 0;
		for (int iAtt = 0; iAtt < data.numAttributes(); iAtt++)
			votesCountByAtt[iAtt] = 0;
		for (int iSample = 0; iSample < numberSamples; iSample++)
			if(attIndexVector[iSample]!=-1){
				votesCountByAtt[attIndexVector[iSample]]++;
				numberVotes++;
			}
		int mostVotedAtt = Utils.maxIndex(votesCountByAtt);

		Distribution checkDistribution = new DistributionConsolidated(samplesVector);
		NoSplit noSplitModel = new NoSplit(checkDistribution);
		// if all nodes are leafs,
		if(numberVotes==0)
			//  return a consolidated leaf
			return noSplitModel;
		
		// Consolidate the split point (if numerical)
		double splitPointConsolidated = consolidateSplitPoint(mostVotedAtt, attIndexVector, splitPointVector, data);
		// Creates the consolidated model
		C45ConsolidatedSplit consolidatedModel =
				new C45ConsolidatedSplit(mostVotedAtt, m_minNoObj, checkDistribution.total(), 
						m_useMDLcorrection, data, samplesVector, splitPointConsolidated);

//		// Set the split point analogue to C45 if attribute numeric.
//		// // It is not necessary for the consolidation process because the median value 
//		// //  is already one of the proposed split points.
//		consolidatedModel.setSplitPoint(data);
		
		if(!consolidatedModel.checkModel())
			return noSplitModel;
		return consolidatedModel;
	}

}
