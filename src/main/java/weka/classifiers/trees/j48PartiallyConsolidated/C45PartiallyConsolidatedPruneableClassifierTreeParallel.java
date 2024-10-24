package weka.classifiers.trees.j48PartiallyConsolidated;

import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.NoSplit;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelectionParallel;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedPruneableClassifierTree;
import weka.core.Instances;
import weka.core.Utils;

public class C45PartiallyConsolidatedPruneableClassifierTreeParallel extends C45PartiallyConsolidatedPruneableClassifierTree {

	private static final long serialVersionUID = -7742907782559492361L;
	
	/** Vector for storing the generated base decision trees
	 *  related to each sample */
	//protected C45PruneableClassifierTreeExtended[] m_sampleTreeVectorParallel;
	
	public C45PartiallyConsolidatedPruneableClassifierTreeParallel(ModelSelection toSelectLocModel,
			C45ModelSelectionExtended baseModelToForceDecision, boolean pruneTree, float cf, boolean raiseTree,
			boolean cleanup, boolean collapseTree, int numberSamples, boolean notPreservingStructure) throws Exception {
		super(toSelectLocModel, baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree, numberSamples,
				notPreservingStructure);
		
		// Initialize each base decision tree of the vector
		ModelSelection modelToConsolidate = ((C45ConsolidatedModelSelectionParallel)toSelectLocModel).getModelToConsolidateParallel();
		m_sampleTreeVector = new C45PruneableClassifierTreeExtended[numberSamples];
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample] = new C45PruneableClassifierTreeExtended(
					modelToConsolidate,	baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree,
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
		buildTreeParallel(data, samplesVector, m_subtreeRaising || !m_cleanup, numCore);
		long endTimeBT = System.nanoTime();
		
		long startTimeCoPr = System.nanoTime();
		if (m_collapseTheTree) {
			collapseParallel();
		}
		if (m_pruneTheTree) {
			pruneParallel();
		}
		long endTimeCoPr = System.nanoTime();
		
		long startTimePC = System.nanoTime();
		leavePartiallyConsolidatedParallel(consolidationPercent, numCore);
		long endTimePC = System.nanoTime();
		
		long startTimeBagging = System.nanoTime();
		applyBaggingParallel(numCore);
		long endTimeBagging = System.nanoTime();
		
		long execTimeBT = (endTimeBT - startTimeBT) / 1000;
		long execTimeCoPr = (endTimeCoPr - startTimeCoPr) / 1000;
		long execTimePC = (endTimePC - startTimePC) / 1000;
		long execTimeBagging = (endTimeBagging - startTimeBagging) / 1000;
		long totalTime = execTimeBT + execTimePC + execTimeBagging;
		
		//System.out.println("Zuhaitzaren eraiketak " + execTimeBT + " us behar izan ditu \n");
		//System.out.println("Zuhaitzaren kolapso eta inausketak " + execTimeCoPr + " us behar izan ditu \n");
		//System.out.println("Kontsolidazio partzialaren exekuzioak " + execTimePC + " us behar izan ditu \n");
		//System.out.println("Bagging-en exekuzioak " + execTimeBagging + " us behar izan ditu \n");
		System.out.println("Exekuzio denbora guztira: " + totalTime + " us \n");
		
		
		if (m_cleanup)
			cleanup(new Instances(data, 0));
	}
	
	/**
	 * Builds the consolidated tree structure concurrently.
	 * (based on the method buildTree() of the class 'ClassifierTree')
	 *
	 * @param data the data for pruning the consolidated tree
	 * @param samplesVector the vector of samples used for consolidation
	 * @param keepData is training data to be kept?
	 * @param numCore the number of threads going to be used to build the tree in parallel
	 * @throws Exception if something goes wrong
	 */
	public void buildTreeParallel(Instances data, Instances[] samplesVector, boolean keepData, int numCore) throws Exception {
		/** Number of Samples. */
		int numberSamples = samplesVector.length;

		/** Initialize the consolidated tree */
		if (keepData) {
			m_train = data;
		}
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		
		ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
		
		//final CountDownLatch doneSignal = new CountDownLatch(numberSamples);
		
		final AtomicInteger numFailed = new AtomicInteger();
		
		/**Initialize the base trees */
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample].initiliazeTree(samplesVector[iSample], keepData);
		
		/** Initialize the base trees concurrently */
		/**for (int iSample = 0; iSample < numberSamples; iSample++) {
			
			final Instances[] currentSamplesVector = samplesVector;
			final int currentSample = iSample;
			
			Runnable initTreeTask = new Runnable() {
				public void run() {
					try {
						m_sampleTreeVector[currentSample].initiliazeTree(currentSamplesVector[currentSample], keepData);
					} catch (Throwable e) {
						e.printStackTrace();
						numFailed.incrementAndGet();
						System.err.println("Iteration " + currentSample + " failed!");
					} finally {
						doneSignal.countDown();
					}
				}
			};
			executorPool.submit(initTreeTask);
		}
		doneSignal.await();**/
	    //executorPool.shutdownNow();

		/** Select the best model to split (if it is worth) based on the consolidation process */
		m_localModel = ((C45ConsolidatedModelSelectionParallel)m_toSelectModel).selectModelParallel(data, samplesVector, numCore);
		
		final CountDownLatch doneSignal2 = new CountDownLatch(numberSamples);
		
		
		for (int iSample = 0; iSample < numberSamples; iSample++) {
			
			final Instances[] currentSamplesVector = samplesVector;
			final int currentSample = iSample;
			
			Runnable setLocalModelTask = new Runnable() {
				public void run() {
					try {
						m_sampleTreeVector[currentSample].setLocalModel(currentSamplesVector[currentSample],m_localModel);
					} catch(Throwable e) {
						e.printStackTrace();
						numFailed.incrementAndGet();
						System.err.println("Iteration " + currentSample + " failed!");
					} finally {
						doneSignal2.countDown();
					}
				}
			};
			executorPool.submit(setLocalModelTask);
		}
		doneSignal2.await(); 
		//executorPool.shutdownNow();

		if (m_localModel.numSubsets() > 1) {
			/** Vector storing the obtained subsamples after the split of data */
			Instances [] localInstances;
			/** Vector storing the obtained subsamples after the split of each sample of the vector */
			//ArrayList<Instances[]> localInstancesVector = new ArrayList<Instances[]>();
			//ensureSize(localInstancesVector, numberSamples);
			
			final Instances[][] localInstancesMatrix = new Instances[numberSamples][];
			
			/** For some base trees, although the current node is not a leaf, it could be empty.
			 *  This is necessary in order to calculate correctly the class membership probabilities
			 *   for the given test instance in each base tree */
			for (int iSample = 0; iSample < numberSamples; iSample++)
				if (Utils.eq(m_sampleTreeVector[iSample].getLocalModel().distribution().total(), 0))
					m_sampleTreeVector[iSample].setIsEmpty(true);
			
			
			/** Split data according to the consolidated m_localModel */
			localInstances = m_localModel.split(data);
			
			//for (int iSample = 0; iSample < numberSamples; iSample++)
				//localInstancesVector.add(m_localModel.split(samplesVector[iSample]));
			
			
			final CountDownLatch doneSignal3 = new CountDownLatch(numberSamples);
			
			final Instances[] currentSamplesVector = samplesVector;
			
			for (int iSample = 0; iSample < numberSamples; iSample++) {
				
				final int nSample = iSample;
				
				Runnable localInstanceAddTask = new Runnable() {
					public void run() {
						try {
							Instances[] currentInstance = m_localModel.split(currentSamplesVector[nSample]);
							localInstancesMatrix[nSample] = currentInstance;
							
						
						} catch(Throwable e) {
							e.printStackTrace();
							numFailed.incrementAndGet();
							System.out.println("Iteration " + nSample + " failed!");
						} finally {
							doneSignal3.countDown();
						}
					}
				};
				executorPool.submit(localInstanceAddTask);
			}
			doneSignal3.await();
			

			/** Create the child nodes of the current node and call recursively to getNewTree() */
			data = null;
			samplesVector = null;
			m_sons = new ClassifierTree [m_localModel.numSubsets()];
			
			final CountDownLatch doneSignal4 = new CountDownLatch(numberSamples);
			
			for (int iSample = 0; iSample < numberSamples; iSample++) {
				
				final int currentSample = iSample;
				
				Runnable createSonsVectorTask = new Runnable() {
					public void run() {
						try {
							((C45PruneableClassifierTreeExtended)m_sampleTreeVector[currentSample]).createSonsVector(m_localModel.numSubsets());
						} catch (Throwable e) {
							e.printStackTrace();
							numFailed.incrementAndGet();
							System.out.println("Iteration " + currentSample + " failed!");
						} finally {
							doneSignal4.countDown();
						}
					}
				};
				executorPool.submit(createSonsVectorTask);
			}
			doneSignal4.await();
			//executorPool.shutdownNow();
			
				
			/** Vector storing the subsamples related to the iSon-th son */
			
			
			/**Instances[] localSamplesVector = new Instances[numberSamples];
			 * 
			 * INNER LOOP PARALLELIZATION (NOT WORKING)
			
			final CountDownLatch doneSignal4 = new CountDownLatch(numberSamples);
			
			for (int iSample = 0; iSample < numberSamples; iSample++) {
				
				final int currentSample = iSample;
				final int currentSon = iSon;
				final Instances[] currentLocalSamplesVector = localSamplesVector;
				
				Runnable localSonTask = new Runnable() {
					public void run() {
						try {
							currentLocalSamplesVector[currentSample] =
									((Instances[]) localInstancesVector.get(currentSample))[currentSon];
						} catch (Throwable e) {
							e.printStackTrace();
							numFailed.incrementAndGet();
							System.out.println("Iteration " + currentSample + "of son " + currentSon + " failed!");
						} finally {
							doneSignal4.countDown();
						}
					}
				};
				executorPool.submit(localSonTask);
			}
			doneSignal4.await();	
			executorPool.shutdownNow();**/
			
			
			
			/** OUTER LOOP PARALLELIZATION (WORKING)**/
			
			final CountDownLatch doneSignal5 = new CountDownLatch(m_sons.length);
			
			final Instances[] currentLocalInstances = localInstances;
			
			for (int iSon = 0; iSon < m_sons.length; iSon++) {
				
				final int currentSon = iSon;
				
				Runnable getNewTreeTask = new Runnable() {
					public void run() {
						try {
							//Vector storing the subsamples related to the iSon-th son
							Instances[] localSamplesVector = new Instances[numberSamples];
							for (int iSample = 0; iSample < numberSamples; iSample++)
								localSamplesVector[iSample] = localInstancesMatrix[iSample][currentSon];
								//localSamplesVector[iSample] =
								//((Instances[]) localInstancesVector.get(iSample))[currentSon];
								
							m_sons[currentSon] = (C45PartiallyConsolidatedPruneableClassifierTree)getNewTreeParallel(
										currentLocalInstances[currentSon], localSamplesVector, m_sampleTreeVector, currentSon, numCore);

							currentLocalInstances[currentSon] = null;
							localSamplesVector = null;
						} catch(Throwable e) {
							e.printStackTrace();
							numFailed.incrementAndGet();
							System.out.println("Iteration " + currentSon + " of getNewTreeTask failed!");
						} finally {
							doneSignal5.countDown();
						}
					}
				};
				executorPool.submit(getNewTreeTask);
			}
			doneSignal5.await();
			
			
			localInstances = null;
			//localInstancesVector.clear();
			//localInstancesMatrix = null;
		}else{
			m_isLeaf = true;
			for (int iSample = 0; iSample < numberSamples; iSample++)
				m_sampleTreeVector[iSample].setIsLeaf(true);

			if (Utils.eq(m_localModel.distribution().total(), 0)){
				m_isEmpty = true;
				for (int iSample = 0; iSample < numberSamples; iSample++)
					m_sampleTreeVector[iSample].setIsEmpty(true);
			}
			data = null;
			samplesVector = null;
		}
		executorPool.shutdownNow();
	}
	
	/**
	 * Returns a newly created tree.
	 *
	 * @param data the data to work with
	 * @param samplesVector the vector of samples for building the consolidated tree
	 * @param numCore the number of threads going to be used to build the tree in parallel
	 * @return the new consolidated tree
	 * @throws Exception if something goes wrong
	 */
	protected ClassifierTree getNewTreeParallel(Instances data, Instances[] samplesVector,
			ClassifierTree[] sampleTreeVectorParent, int iSon, int numCore) throws Exception {
		/** Number of Samples. */
		int numberSamples = samplesVector.length;

		C45ModelSelectionExtended baseModelToForceDecision = m_sampleTreeVector[0].getBaseModelToForceDecision();
		C45PartiallyConsolidatedPruneableClassifierTreeParallel newTree =
				new C45PartiallyConsolidatedPruneableClassifierTreeParallel(m_toSelectModel, baseModelToForceDecision,
						m_pruneTheTree, m_CF, m_subtreeRaising, m_cleanup, m_collapseTheTree , samplesVector.length,
						m_pruneBaseTreesWithoutPreservingConsolidatedStructure);
		/** Set the recent created base trees like the sons of the given parent node */
		for (int iSample = 0; iSample < numberSamples; iSample++)
			((C45PruneableClassifierTreeExtended)sampleTreeVectorParent[iSample]).setIthSon(iSon, newTree.m_sampleTreeVector[iSample]);
		newTree.buildTreeParallel(data, samplesVector, m_subtreeRaising, numCore);

		return newTree;
	}
	
	/**
	 * Collapses a tree to a node if training error doesn't increase.
	 * And also each tree of a tree vector to a node.
	 * 
	 *  @throws Exception if something goes wrong
	 */
	public final void collapseParallel() throws Exception{
	
		double errorsOfSubtree;
		double errorsOfTree;
		int i;
		
		//ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
		
		//final CountDownLatch doneSignal = new CountDownLatch(m_sons.length);
		
		//final AtomicInteger numFailed = new AtomicInteger();
	
		if (!m_isLeaf){
			errorsOfSubtree = getTrainingErrors();
			errorsOfTree = localModel().distribution().numIncorrect();
			if (errorsOfSubtree >= errorsOfTree-1E-3)
				setAsLeaf();
			else
				for (i=0;i<m_sons.length;i++)
					son(i).collapse();
				/**for (i=0;i<m_sons.length;i++) {
					
					final int currentSon = i;
					
					Runnable collapseTask = new Runnable() {
						public void run() {
							try {
								son(currentSon).collapse();
							} catch(Throwable e) {
								e.printStackTrace();
								numFailed.incrementAndGet();
								System.out.println("Iteration " + currentSon + " failed!");
							} finally {
								doneSignal.countDown();
							}
						}
					};
					executorPool.submit(collapseTask);
				}
			doneSignal.await();**/
		}
	}
	
	/**
	 * Prunes the consolidated tree using C4.5's pruning procedure, and all base trees in the same way
	 *
	 * @throws Exception if something goes wrong
	 */
	public void pruneParallel() throws Exception {

		double errorsLargestBranch;
		double errorsLeaf;
		double errorsTree;
		int indexOfLargestBranch;
		C45PartiallyConsolidatedPruneableClassifierTreeParallel largestBranch;
		int i;
		
		//ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
		
		//final CountDownLatch doneSignal = new CountDownLatch(m_sons.length);
		
		//final AtomicInteger numFailed = new AtomicInteger();

		if (!m_isLeaf){

			// Prune all subtrees.
			for (i=0;i<m_sons.length;i++)
				son(i).prune();
			
			/**for (i=0;i<m_sons.length;i++) {
				
				final int currentSon = i;
				
				Runnable sonPruneTask = new Runnable() {
					public void run() {
						try {
							son(currentSon).prune();
						} catch(Throwable e) {
							e.printStackTrace();
							numFailed.incrementAndGet();
							System.out.println("Iteration " + currentSon + " failed!");
						} finally {
							doneSignal.countDown();
						}
					}
				};
				executorPool.submit(sonPruneTask);
			}
			doneSignal.await();**/
				
			// Compute error for largest branch
			indexOfLargestBranch = localModel().distribution().maxBag();
			if (m_subtreeRaising) {
				errorsLargestBranch = ((C45PartiallyConsolidatedPruneableClassifierTreeParallel)(son(indexOfLargestBranch))).
						getEstimatedErrorsForBranch((Instances)m_train);
			} else {
				errorsLargestBranch = Double.MAX_VALUE;
			}

			// Compute error if this Tree would be leaf
			errorsLeaf = 
					getEstimatedErrorsForDistribution(localModel().distribution());

			// Compute error for the whole subtree
			errorsTree = getEstimatedErrors();

			// Decide if leaf is best choice.
			if (Utils.smOrEq(errorsLeaf,errorsTree+0.1) &&
					Utils.smOrEq(errorsLeaf,errorsLargestBranch+0.1)){
				setAsLeaf();
				return;
			}

			// Decide if largest branch is better choice
			// than whole subtree.
			if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)){
				largestBranch = (C45PartiallyConsolidatedPruneableClassifierTreeParallel)son(indexOfLargestBranch);
				m_sons = largestBranch.m_sons;
				m_localModel = largestBranch.localModel();
				m_isLeaf = largestBranch.m_isLeaf;
				newDistribution(m_train);
				
				// Replace current node with the largest branch in all base trees
				for (int iSample=0; iSample < m_sampleTreeVector.length; iSample++)
					m_sampleTreeVector[iSample].replaceWithIthSubtree(indexOfLargestBranch);
				pruneParallel();
				
				
				/**final CountDownLatch doneSignal2 = new CountDownLatch(m_sampleTreeVector.length);
				for (int iSample=0; iSample < m_sampleTreeVector.length; iSample++) {
					
					final int currentSample = iSample;
					
					Runnable replaceSubtreeTask = new Runnable() {
						public void run() {
							try {
								m_sampleTreeVector[currentSample].replaceWithIthSubtree(indexOfLargestBranch);
							} catch(Throwable e) {
								e.printStackTrace();
								numFailed.incrementAndGet();
								System.out.println("Iteration " + currentSample + " failed!");
							} finally {
								doneSignal2.countDown();
							}
						}
					};
					executorPool.submit(replaceSubtreeTask);
				}
				doneSignal2.await();**/
				
			}
		}
	}
	
	
	
	/**
	 * Prunes the consolidated tree and all base trees according consolidationPercent concurrently.
	 *
	 * @param consolidationPercent percentage of the structure of the tree to leave without pruning 
	 * @param numCore the number of threads going to be used leave the tree partially consolidated
	 * @throws Exception if something goes wrong
	 */
	public void leavePartiallyConsolidatedParallel(float consolidationPercent, int numCore) throws Exception {
		// Number of internal nodes of the consolidated tree
		int innerNodes = numNodes() - numLeaves();
		// Number of nodes of the consolidated tree to leave as consolidated based on given consolidationPercent 
		int numberNodesConso = (int)(((innerNodes * consolidationPercent) / 100) + 0.5);
		setNumInternalNodesConso(numberNodesConso);
		System.out.println("Number of nodes to leave as consolidated: " + numberNodesConso + " of " + innerNodes);
		// Vector storing the nodes to maintain as consolidated
		ArrayList<C45PartiallyConsolidatedPruneableClassifierTreeParallel> nodesConsoVector = new ArrayList<C45PartiallyConsolidatedPruneableClassifierTreeParallel>();
		// Vector storing the weight of the nodes of nodesConsoVector
		ArrayList<Double> weightNodesConsoVector = new ArrayList<Double>(); 
		// Counter of the current number of nodes left as consolidated
		int countNodesConso = 0;
		
		/** Initialize the vectors with the root node (if it has children) */
		if(!m_isLeaf){
			nodesConsoVector.add(this);
			weightNodesConsoVector.add(localModel().distribution().total());
		}
		
		ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
		
		/** Determine which nodes will be left as consolidated according to their weight 
		 *   starting from the root node */
		while((nodesConsoVector.size() > 0) && (countNodesConso < numberNodesConso)){
			/** Add the heaviest node */
			// Look for the heaviest node
			Double valHeaviest = Collections.max(weightNodesConsoVector);
			int iHeaviest = weightNodesConsoVector.indexOf(valHeaviest);
			C45PartiallyConsolidatedPruneableClassifierTreeParallel heaviestNode = nodesConsoVector.get(iHeaviest); 
			// Add the children of the chosen node to the vectors (ONLY if each child is an internal node)
			// // By construction it's guaranteed that heaviestNode has children
			for(int iSon = 0; iSon < heaviestNode.m_sons.length; iSon++)
				if(!(((C45PartiallyConsolidatedPruneableClassifierTreeParallel)heaviestNode.son(iSon)).m_isLeaf)){
					C45PartiallyConsolidatedPruneableClassifierTreeParallel localSon = (C45PartiallyConsolidatedPruneableClassifierTreeParallel)heaviestNode.son(iSon); 
					nodesConsoVector.add(localSon);
					weightNodesConsoVector.add(localSon.localModel().distribution().total());
				}
			// Remove the heaviest node of the vectors
			nodesConsoVector.remove(iHeaviest);
			weightNodesConsoVector.remove(iHeaviest);
			// Increase the counter of consolidated nodes
			countNodesConso++;
		}
		
		final CountDownLatch pruningDoneSignal = new CountDownLatch(nodesConsoVector.size());
		
		final AtomicInteger numFailed = new AtomicInteger();
		
		/** Prune the rest of nodes (also on the base trees)*/
		for(int iNode = 0; iNode < nodesConsoVector.size(); iNode++) {
			
			final int currentNode = iNode;
			
			Runnable PruningTask = new Runnable() {
				public void run() {
					try {
						((C45PartiallyConsolidatedPruneableClassifierTreeParallel)nodesConsoVector.get(currentNode)).setAsLeaf();
					} catch (Throwable e) {
						e.printStackTrace();
						numFailed.incrementAndGet();
						System.out.println("Iteration " + currentNode + " failed!");
					} finally {
						pruningDoneSignal.countDown();
					}
				}
			};
			executorPool.submit(PruningTask);
		}
		pruningDoneSignal.await();
	    executorPool.shutdownNow();
			
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
			
			Runnable BaggingTask = new Runnable() {
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
			executorPool.submit(BaggingTask);
		}
		doneSignal.await();
	    executorPool.shutdownNow();
	}
	
	/**public static void ensureSize(ArrayList<?> list, int size) {
	    // Prevent excessive copying while we're adding
	    list.ensureCapacity(size);
	    while (list.size() < size) {
	        list.add(null);
	    }
	}**/
	


}
