package weka.classifiers.trees.j48PartiallyConsolidated;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;

public class C45PartiallyConsolidatedPruneableClassifierTreeParallel extends C45PartiallyConsolidatedPruneableClassifierTree {

	private static final long serialVersionUID = -7742907782559492361L;
	
	public C45PartiallyConsolidatedPruneableClassifierTreeParallel(ModelSelection toSelectLocModel,
			C45ModelSelectionExtended baseModelToForceDecision, boolean pruneTree, float cf, boolean raiseTree,
			boolean cleanup, boolean collapseTree, int numberSamples, boolean notPreservingStructure) throws Exception {
		super(toSelectLocModel, baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree, numberSamples,
				notPreservingStructure);
	}
	
	

	/**
	 * Method for building a pruneable classifier consolidated tree concurrently.
	 *
	 * @param data the data for pruning the consolidated tree
	 * @param samplesVector the vector of samples for building the consolidated tree
	 * @param consolidationPercent the value of consolidation percent
	 * @param numCore the number of threads going to be used to build the classifier in parallel
	 * @throws Exception if something goes wrong
	 */
	public void buildClassifierParallel(Instances data, Instances[] samplesVector, float consolidationPercent, int numCore) throws Exception {

		long startTimeBT = System.nanoTime();
		buildTree(data, samplesVector, m_subtreeRaising || !m_cleanup);
		long endTimeBT = System.nanoTime();
		if (m_collapseTheTree) {
			collapse();
		}
		if (m_pruneTheTree) {
			prune();
		}
		long startTimePC = System.nanoTime();
		leavePartiallyConsolidated(consolidationPercent);
		long endTimePC = System.nanoTime();
		
		long startTimeBagging = System.nanoTime();
		applyBaggingParallel(numCore);
		long endTimeBagging = System.nanoTime();
		
		long execTimeBT = (endTimeBT - startTimeBT) / 1000;
		long execTimePC = (endTimePC - startTimePC) / 1000;
		long execTimeBagging = (endTimeBagging - startTimeBagging) / 1000;
		
		System.out.println("Zuhaitzaren eraiketak " + execTimeBT + " mikros behar izan ditu \n");
		System.out.println("Kontsolidazio partzialaren exekuzioak " + execTimePC + " mikros behar izan ditu \n");
		System.out.println("Bagging-en exekuzioak " + execTimeBagging + " mikros behar izan ditu \n");
		
		if (m_cleanup)
			cleanup(new Instances(data, 0));
	}
	
	/**
	 * Rebuilds each base tree according to J48 algorithm independently and
	 *  maintaining the consolidated tree structure concurrently
	 *  @param numCore the number of threads going to be used to apply Bagging in parallel
	 * @throws Exception if something goes wrong
	 */
	protected void applyBaggingParallel(int numCore) throws Exception {
		/** Number of Samples. */
		int numberSamples = m_sampleTreeVector.length;
		
		ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
		
		final CountDownLatch doneSignal = new CountDownLatch(numberSamples);
		
		final AtomicInteger numFailed = new AtomicInteger();
		
		
		for (int iSample = 0; iSample < numberSamples; iSample++) {
			
			final int currentSample = iSample;
			
			Runnable newTask = new Runnable() {
				public void run() {
					try {
						m_sampleTreeVector[currentSample].rebuildTreeFromConsolidatedStructureAndPrune();
					} catch (Throwable e) {
						e.printStackTrace();
						numFailed.incrementAndGet();
						System.err.println("Iteration " + currentSample + " failed!");
					} finally {
						doneSignal.countDown();
					}
				}
			};
			executorPool.submit(newTask);
		}
		doneSignal.await();
	    executorPool.shutdownNow();
	}
	
	

}
