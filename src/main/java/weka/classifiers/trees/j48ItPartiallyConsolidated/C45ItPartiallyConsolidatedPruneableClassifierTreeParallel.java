package weka.classifiers.trees.j48ItPartiallyConsolidated;

import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import weka.classifiers.trees.J48It;
import weka.classifiers.trees.J48ItPartiallyConsolidated;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelectionParallel;
import weka.classifiers.trees.j48PartiallyConsolidated.*;
import weka.core.Instances;
import weka.core.Utils;

public class C45ItPartiallyConsolidatedPruneableClassifierTreeParallel extends C45ItPartiallyConsolidatedPruneableClassifierTree {


	private static final long serialVersionUID = -3300820617594466879L;
	
	protected C45ModelSelectionExtended m_baseModelToForceDecision;
	
	
	/**
	 * Constructor for pruneable consolidated tree structure. Calls the superclass
	 * constructor.
	 *
	 * @param toSelectLocModel      selection method for local splitting model
	 * @param pruneTree             true if the tree is to be pruned
	 * @param cf                    the confidence factor for pruning
	 * @param raiseTree             true if subtree raising has to be performed
	 * @param cleanup               true if cleanup has to be done
	 * @param collapseTree          true if collapse has to be done
	 * @param numberSamples         Number of Samples
	 * @param ITPCTmaximumCriteria  maximum number of nodes or levels
	 * @param ITPCTpriorityCriteria criteria to build the tree
	 * @param pruneCT true if the CT tree is to be pruned
	 * @param collapseCT true if the CT tree is to be collapsed
	 * @throws Exception if something goes wrong
	 */
	public C45ItPartiallyConsolidatedPruneableClassifierTreeParallel(ModelSelection toSelectLocModel,
			C45ModelSelectionExtended baseModelToForceDecision, boolean pruneTree, float cf, boolean raiseTree,
			boolean cleanup, boolean collapseTree, int numberSamples, boolean notPreservingStructure,
			int ITPCTpriorityCriteria, boolean pruneCT, boolean collapseCT) throws Exception {
		super(toSelectLocModel, baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree, numberSamples,
				notPreservingStructure, ITPCTpriorityCriteria, pruneCT, collapseCT);
		
		m_baseModelToForceDecision = baseModelToForceDecision;
	}
	
	
	/**
	 * Method for building a pruneable classifier consolidated tree.
	 *
	 * @param data                 the data for pruning the consolidated tree
	 * @param samplesVector        the vector of samples for building the
	 *                             consolidated tree
	 * @param consolidationPercent the value of consolidation percent
	 * @param numCore the number of threads going to be used to build the classifier in parallel
	 * @throws Exception if something goes wrong
	 */
	public void buildClassifierParallel(Instances data, Instances[] samplesVector, float consolidationPercent,
			int consolidationNumberHowToSet, int numCore) throws Exception {
		long trainTimeStart = 0, trainTimeElapsed = 0;
		
		
		if (m_priorityCriteria == J48It.Original) {

			m_pruneTheTree = m_pruneTheConsolidatedTree;
			m_collapseTheTree = m_collapseTheCTree;
			C45PartiallyConsolidatedPruneableClassifierTreeParallel originalParallelTree = new C45PartiallyConsolidatedPruneableClassifierTreeParallel(m_toSelectModel, 
					this.m_baseModelToForceDecision, this.m_pruneTheTree, this.m_CF, this.m_subtreeRaising, this.m_cleanup, this.m_collapseTheTree, 
					samplesVector.length, this.m_pruneBaseTreesWithoutPreservingConsolidatedStructure);
			//super.buildClassifier(data, samplesVector, consolidationPercent);
			originalParallelTree.buildClassifierParallel(data, samplesVector, consolidationPercent, numCore);
			
			

		} else {
			if (consolidationNumberHowToSet == J48ItPartiallyConsolidated.ConsolidationNumber_Percentage) {
								
				trainTimeStart = System.currentTimeMillis();
				
				C45PartiallyConsolidatedPruneableClassifierTreeParallel originalParallelTree = new C45PartiallyConsolidatedPruneableClassifierTreeParallel(m_toSelectModel, 
						this.m_baseModelToForceDecision, this.m_pruneTheTree, this.m_CF, this.m_subtreeRaising, this.m_cleanup, this.m_collapseTheTree, 
						samplesVector.length, this.m_pruneBaseTreesWithoutPreservingConsolidatedStructure);
				
				originalParallelTree.buildTreeParallel(data, samplesVector, m_subtreeRaising || !m_cleanup, numCore); // build the tree without restrictions
								
				if (m_collapseTheCTree) {
					originalParallelTree.collapseParallel();
				}
				if (m_pruneTheConsolidatedTree) {
					originalParallelTree.pruneParallel();
				}
				trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
				System.out.println("Time taken to build the whole consolidated tree: " + Utils.doubleToString(trainTimeElapsed, 2) + " milliseconds\n");
				//System.out.println("Time taken to build the whole consolidated tree: " + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds\n");
				m_elapsedTimeTrainingWholeCT = trainTimeElapsed / (double)1000.0;

				if (m_priorityCriteria == J48It.Levelbylevel) {

					// Number of levels of the consolidated tree
					int treeLevels = numLevels();

					// Number of levels of the consolidated tree to leave as consolidated based on
					// given consolidationPercent
					int numberLevelsConso = (int) (((treeLevels * consolidationPercent) / 100) + 0.5);
					m_maximumCriteria = numberLevelsConso;
					setNumInternalNodesConso(numberLevelsConso);
					System.out.println(
							"Number of levels to leave as consolidated: " + numberLevelsConso + " of " + treeLevels);

				} else {

					// Number of internal nodes of the consolidated tree
					int innerNodes = numNodes() - numLeaves();

					// Number of nodes of the consolidated tree to leave as consolidated based on
					// given consolidationPercent
					int numberNodesConso = (int) (((innerNodes * consolidationPercent) / 100) + 0.5);
					m_maximumCriteria = numberNodesConso;
					setNumInternalNodesConso(numberNodesConso);
					System.out.println(
							"Number of nodes to leave as consolidated: " + numberNodesConso + " of " + innerNodes);

				}

			} else // consolidationNumberHowToSet ==
					// J48ItPartiallyConsolidated.ConsolidationNumber_Value
			{
				m_maximumCriteria = (int) consolidationPercent;
				System.out.println("Number of nodes or levels to leave as consolidated: " + m_maximumCriteria);
				m_elapsedTimeTrainingWholeCT = (double)0.0;
			}

			// buildTree
			trainTimeStart = System.currentTimeMillis();
			buildTreeParallel(data, samplesVector, m_subtreeRaising || !m_cleanup, numCore);
			if (m_collapseTheCTree) {
				collapse();
			}
			if (m_pruneTheConsolidatedTree) {
				prune();
			}
			trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
			//System.out.println("Time taken to build the partial consolidated tree: " + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds\n");
			System.out.println("Time taken to build the partial consolidated tree: " + Utils.doubleToString(trainTimeElapsed, 2) + " milliseconds\n");
			m_elapsedTimeTrainingPartialCT = trainTimeElapsed / (double)1000.0;

			trainTimeStart = System.currentTimeMillis();
			applyBagging();
			trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
			//System.out.println("Time taken to build the associated Bagging: " + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds\n");
			System.out.println("Time taken to build the associated Bagging: " + Utils.doubleToString(trainTimeElapsed, 2) + " milliseconds\n");
			m_elapsedTimeTrainingAssocBagging = trainTimeElapsed / (double)1000.0;

			if (m_cleanup)
				cleanup(new Instances(data, 0));
		}
	}
	
	
	/**
	 * Builds the consolidated tree structure concurrently. (based on the method buildTree() of
	 * the class 'ClassifierTree')
	 *
	 * @param data          the data for pruning the consolidated tree
	 * @param samplesVector the vector of samples used for consolidation
	 * @param keepData      is training data to be kept?
	 * @param numCore the number of threads going to be used to build the tree in parallel
	 * @throws Exception if something goes wrong
	 */
	public void buildTreeParallel(Instances data, Instances[] samplesVector, boolean keepData, int numCore) throws Exception {
		/** Number of Samples. */
		int numberSamples = samplesVector.length;
		
		/**Initialize the threads**/
		ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
		final AtomicInteger numFailed = new AtomicInteger();

		/** Initialize the consolidated tree */
		if (keepData) {
			m_train = data;
		}
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		/** Initialize the base trees */
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample].initiliazeTree(samplesVector[iSample], keepData);

		ArrayList<Object[]> list = new ArrayList<>();

		// add(Data, samplesVector, tree, orderValue, currentLevel)
		list.add(new Object[] { data, samplesVector, this, null, 0 }); // The parent node is considered level 0

		int index = 0;
		//double orderValue;

		int internalNodes = 0;

		while (list.size() > 0) {

			Object[] current = list.get(0);

			/** Current node level. **/
			int currentLevel = (int) current[4];

			/** Number of Samples. */
			Instances[] currentSamplesVector = (Instances[]) current[1];
			//int numberSamples = currentSamplesVector.length;

			list.set(0, null); // Null to free up memory
			list.remove(0);

			Instances currentData = (Instances) current[0];
			C45ItPartiallyConsolidatedPruneableClassifierTreeParallel currentTree = (C45ItPartiallyConsolidatedPruneableClassifierTreeParallel) current[2];
			currentTree.m_order = index;

			/** Initialize the consolidated tree */
			if (keepData) {
				currentTree.m_train = currentData;
			}
			currentTree.m_test = null;
			currentTree.m_isLeaf = false;
			currentTree.m_isEmpty = false;
			currentTree.m_sons = null;

			/** Initialize the base trees */
			for (int iSample = 0; iSample < numberSamples; iSample++)
				currentTree.m_sampleTreeVector[iSample].initiliazeTree(currentSamplesVector[iSample], keepData);

			/**
			 * Select the best model to split (if it is worth) based on the consolidation
			 * process
			 */
			currentTree.m_localModel = ((C45ConsolidatedModelSelectionParallel) currentTree.m_toSelectModel)
					.selectModelParallel(currentData, currentSamplesVector, numCore);
			
			final CountDownLatch doneSignal = new CountDownLatch(numberSamples);
			
			for (int iSample = 0; iSample < numberSamples; iSample++) {
				
				final Instances[] currentParallelSamplesVector = currentSamplesVector;
				final int currentSample = iSample;
				
				Runnable setLocalModelTask = new Runnable() {
					public void run() {
						try {
							currentTree.m_sampleTreeVector[currentSample].setLocalModel(currentParallelSamplesVector[currentSample],
									currentTree.m_localModel);
						} catch(Throwable e) {
							e.printStackTrace();
							numFailed.incrementAndGet();
							System.out.println("Iteration " + currentSample + " failed!");
						} finally {
							doneSignal.countDown();
						}
					}
				};
				executorPool.submit(setLocalModelTask);
			}
			doneSignal.await();
				

			if ((currentTree.m_localModel.numSubsets() > 1) && ((m_priorityCriteria == J48It.Original)
					|| ((m_priorityCriteria == J48It.Levelbylevel) && (currentLevel < m_maximumCriteria))
					|| ((m_priorityCriteria > J48It.Levelbylevel) && (internalNodes < m_maximumCriteria)))) {

				/** Vector storing the obtained subsamples after the split of data */
				Instances[] localInstances;
				/**
				 * Vector storing the obtained subsamples after the split of each sample of the
				 * vector
				 */
				ArrayList<Instances[]> localInstancesVector = new ArrayList<Instances[]>();

				/**
				 * For some base trees, although the current node is not a leaf, it could be
				 * empty. This is necessary in order to calculate correctly the class membership
				 * probabilities for the given test instance in each base tree
				 */

				ArrayList<Object[]> listSons = new ArrayList<>();

				for (int iSample = 0; iSample < numberSamples; iSample++)
					if (Utils.eq(currentTree.m_sampleTreeVector[iSample].getLocalModel().distribution().total(), 0))
						currentTree.m_sampleTreeVector[iSample].setIsEmpty(true);

				/** Split data according to the consolidated m_localModel */
				localInstances = currentTree.m_localModel.split(currentData);
				for (int iSample = 0; iSample < numberSamples; iSample++)
					localInstancesVector.add(currentTree.m_localModel.split(currentSamplesVector[iSample]));

				/**
				 * Create the child nodes of the current node and call recursively to
				 * getNewTree()
				 */
				currentData = null;
				currentSamplesVector = null;
				currentTree.m_sons = new ClassifierTree[currentTree.m_localModel.numSubsets()];
				
				final CountDownLatch doneSignal2 = new CountDownLatch(numberSamples);
				
				for (int iSample = 0; iSample < numberSamples; iSample++) {
					
					final int currentSample = iSample;
					
					Runnable createSonsVectorTask = new Runnable() {
						public void run() {
							try {
								((C45PruneableClassifierTreeExtended) currentTree.m_sampleTreeVector[currentSample])
								.createSonsVector(currentTree.m_localModel.numSubsets());
							} catch(Throwable e) {
								e.printStackTrace();
								numFailed.incrementAndGet();
								System.out.println("Iteration " + currentSample + " failed!");
							} finally {
								doneSignal2.countDown();
							}
						}
					};
					executorPool.submit(createSonsVectorTask);
				}
				doneSignal2.await();
					

				//////////////////
				C45ModelSelectionExtended baseModelToForceDecision = 
						currentTree.m_sampleTreeVector[0].getBaseModelToForceDecision();
				
				final CountDownLatch doneSignal3 = new CountDownLatch(m_sons.length);
				
				for (int iSon = 0; iSon < currentTree.m_sons.length; iSon++) {
					
					final ArrayList<Object[]> currentList = list;
					final ArrayList<Object[]> currentListSons = listSons;
					final Instances[] currentLocalInstances = localInstances;
					final int currentSon = iSon;
					
					Runnable getNewTreeTask = new Runnable() {
						public void run() {
							try {
								/** Vector storing the subsamples related to the iSon-th son */
								Instances[] localSamplesVector = new Instances[numberSamples];
								for (int iSample = 0; iSample < numberSamples; iSample++)
									localSamplesVector[iSample] = ((Instances[]) localInstancesVector.get(iSample))[currentSon];

								// getNewTree
								C45ItPartiallyConsolidatedPruneableClassifierTreeParallel newTree = new C45ItPartiallyConsolidatedPruneableClassifierTreeParallel(
										currentTree.m_toSelectModel, baseModelToForceDecision, m_pruneTheTree, m_CF,
										m_subtreeRaising, m_cleanup, m_collapseTheTree, localSamplesVector.length,
										m_pruneBaseTreesWithoutPreservingConsolidatedStructure,
										m_priorityCriteria, m_pruneTheConsolidatedTree, m_collapseTheCTree);

								/** Set the recent created base trees like the sons of the given parent node */
								for (int iSample = 0; iSample < numberSamples; iSample++)
									((C45PruneableClassifierTreeExtended) currentTree.m_sampleTreeVector[iSample]).setIthSon(currentSon,
											newTree.m_sampleTreeVector[iSample]);
								
								double orderValue;

								if (m_priorityCriteria == J48ItPartiallyConsolidated.Size) // Added by size, largest to smallest
								{

									orderValue = currentTree.m_localModel.distribution().perBag(currentSon);

									Object[] son = new Object[] { currentLocalInstances[currentSon], localSamplesVector, newTree, orderValue,
											currentLevel + 1 };
									addSonOrderedByValue(currentList, son);

								} else if (m_priorityCriteria == J48ItPartiallyConsolidated.Gainratio) // Added by gainratio,
																										// largest to smallest
								{
									ClassifierSplitModel sonModel = ((C45ItPartiallyConsolidatedPruneableClassifierTreeParallel) newTree).m_toSelectModel
											.selectModel(currentLocalInstances[currentSon]);
									if (sonModel.numSubsets() > 1) {

										orderValue = ((C45Split) sonModel).gainRatio();

									} else {

										orderValue = (double) Double.MIN_VALUE;
									}
									Object[] son = new Object[] { currentLocalInstances[currentSon], localSamplesVector, newTree, orderValue,
											currentLevel + 1 };
									addSonOrderedByValue(currentList, son);

								} else if (m_priorityCriteria == J48ItPartiallyConsolidated.Gainratio_normalized) // Added by
																													// gainratio
																													// normalized,
								// largest to smallest
								{

									double size = currentTree.m_localModel.distribution().perBag(currentSon);
									double gainRatio;
									ClassifierSplitModel sonModel = ((C45ItPartiallyConsolidatedPruneableClassifierTreeParallel) newTree).m_toSelectModel
											.selectModel(currentLocalInstances[currentSon]);
									if (sonModel.numSubsets() > 1) {

										gainRatio = ((C45Split) sonModel).gainRatio();
										orderValue = size * gainRatio;

									} else {

										orderValue = (double) Double.MIN_VALUE;
									}
									Object[] son = new Object[] { currentLocalInstances[currentSon], localSamplesVector, newTree, orderValue,
											currentLevel + 1 };
									addSonOrderedByValue(currentList, son);

								} else {
									currentListSons.add(new Object[] { currentLocalInstances[currentSon], localSamplesVector, newTree, 0,
											currentLevel + 1 });
								}

								currentTree.m_sons[currentSon] = newTree;

								currentLocalInstances[currentSon] = null;
								localSamplesVector = null;
							} catch(Throwable e) {
								e.printStackTrace();
								numFailed.incrementAndGet();
								System.out.println("Iteration " + currentSon + " failed!");
							} finally {
								doneSignal3.countDown();
							}
						}
					};
					executorPool.submit(getNewTreeTask);
				}
				doneSignal3.await();

				if (m_priorityCriteria == J48ItPartiallyConsolidated.Levelbylevel) { // Level by level
					list.addAll(listSons);
				}

				else if (m_priorityCriteria == J48ItPartiallyConsolidated.Preorder
						|| m_priorityCriteria == J48ItPartiallyConsolidated.Original) { // Preorder
					listSons.addAll(list);
					list = listSons;
				}

				localInstances = null;
				localInstancesVector.clear();
				listSons = null;
				internalNodes++;

			} else {
				currentTree.m_isLeaf = true;
				for (int iSample = 0; iSample < numberSamples; iSample++)
					currentTree.m_sampleTreeVector[iSample].setIsLeaf(true);

				if (Utils.eq(currentTree.m_localModel.distribution().total(), 0)) {
					currentTree.m_isEmpty = true;
					for (int iSample = 0; iSample < numberSamples; iSample++)
						currentTree.m_sampleTreeVector[iSample].setIsEmpty(true);
				}
				currentData = null;
				currentSamplesVector = null;
			}
			index++;

		}
		executorPool.shutdownNow();
	}
	

}
