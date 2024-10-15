package weka.classifiers.trees;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelectionParallel;
import weka.classifiers.trees.j48Consolidated.InstancesConsolidated;
import weka.classifiers.trees.j48PartiallyConsolidated.C45ModelSelectionExtended;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PartiallyConsolidatedPruneableClassifierTree;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PartiallyConsolidatedPruneableClassifierTreeParallel;
import weka.core.AdditionalMeasureProducer;
import weka.core.Drawable;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Summarizable;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class J48PartiallyConsolidatedParallel 
	extends J48PartiallyConsolidated 
	implements OptionHandler, Drawable, Matchable, Sourcable,
	WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer,
	TechnicalInformationHandler {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = -1119606846120036468L;
	protected int m_numExecutionSlots = 1;
	
	public void setNumExecutionSlots(int numSlots) {
		this.m_numExecutionSlots = numSlots;
	}

	public int getNumExecutionSlots() {
		return this.m_numExecutionSlots;
	}

	public String numExecutionSlotsTipText() {
		return "The number of execution slots (threads) to use for constructing the ensemble.";
	}
	
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector(2);
		newVector.addElement(new Option(
				"\tNumber of execution slots.\n\t(default 1 - i.e. no parallelism)\n\t(use 0 to auto-detect number of cores)",
				"num-slots", 1, "-num-slots <num>"));
		newVector.addAll(Collections.list(super.listOptions()));
		return newVector.elements();
	}
	
	public void setOptions(String[] options) throws Exception {
		String iterations = Utils.getOption("num-slots", options);
		if (iterations.length() != 0) {
			this.setNumExecutionSlots(Integer.parseInt(iterations));
		} else {
			this.setNumExecutionSlots(1);
		}

		super.setOptions(options);
	}
	
	public String[] getOptions() {
		String[] superOptions = super.getOptions();
		String[] options = new String[superOptions.length + 2];
		int current = 0;
		options[current++] = "-num-slots";
		options[current++] = "" + this.getNumExecutionSlots();
		System.arraycopy(superOptions, 0, options, current, superOptions.length);
		return options;
	}
	
	/**
	 * Klasifikatzailea eraikitzen du seriean / Builds classifier serially 
	 * (J48PartiallyConsolidated klase gurasoko buildClassifiers() metodoa deitzen du)
	 * 
	 * @throws Exception if number of execution slots is lower than 0
	 */
	/*public void buildClassifier(Instances instances) throws Exception {
		super.buildClassifier(instances);
		if (this.m_numExecutionSlots < 0) {
			throw new Exception("Number of execution slots needs to be >= 0!");
		}
	}*/
	
	/**
	 * Klasifikatzailea eraikitzen du paraleloan / Builds classifier concurrently
	 * 
	 * @param instances the data to train the classifier with 
	 * @throws Exception if classifier can't be built successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {
		int numCore;
		
		if (this.m_numExecutionSlots < 0) {
			System.out.println("ERROR!!!");
			throw new Exception("Number of execution slots needs to be >= 0!");
		}
		
		if (this.m_numExecutionSlots != 1) {
			numCore = this.m_numExecutionSlots == 0 ? Runtime.getRuntime().availableProcessors() : this.m_numExecutionSlots;
			
			// can classifier tree handle the data?
			getCapabilities().testWithFail(instances);
			
			// remove instances with missing class before generate samples
			instances = new Instances(instances);
			instances.deleteWithMissingClass();
			
			long startTime = System.nanoTime();
			//Generate as many samples as the number of samples with the given instances
			Instances[] samplesVector = super.generateSamples(instances);
			//Instances[] samplesVector = generateSamplesParallel(instances, numCore);
			long endTime = System.nanoTime();
			
			long execTime = (endTime - startTime) / 1000;
			
			//System.out.println("Laginketak " + execTime + " mikros behar izan ditu \n");
			
			//System.out.println("\n " + samplesVector.length + " \n" );
		    //if (m_Debug) printSamplesVector(samplesVector);
			/** Set the model selection method to determine the consolidated decisions */
		    ModelSelection modSelection;
			// TODO Implement the option binarySplits of J48
			modSelection = new C45ConsolidatedModelSelectionParallel(m_minNumObj, instances, 
					m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
			/** Set the model selection method to force the consolidated decision in each base tree*/
			C45ModelSelectionExtended baseModelToForceDecision = new C45ModelSelectionExtended(m_minNumObj, instances, 
					m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
			// TODO Implement the option reducedErrorPruning of J48
			C45PartiallyConsolidatedPruneableClassifierTreeParallel localClassifier =
					new C45PartiallyConsolidatedPruneableClassifierTreeParallel(modSelection, baseModelToForceDecision,
							!m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup, m_collapseTree, samplesVector.length,
							m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure);
			
			/*for (int i=0; i < samplesVector.length; i++) {
				if (samplesVector[i] == null) {
					System.out.println("i hau NULL da: " + i + "\n");
				} else {
					System.out.println(samplesVector[i].toString());
				}
				
			}*/
			//System.out.println("samplesVector length: " + samplesVector.length + "\n");

			localClassifier.buildClassifierParallel(instances, samplesVector, m_PCTBconsolidationPercent, numCore);

			m_root = localClassifier;
			m_Classifiers = localClassifier.getSampleTreeVector();
			// // We could get any base tree of the vector as root and use it in the graphical interface
			// // (for example, to visualize it)
			// //m_root = localClassifier.getSampleTreeIth(0);
			
			((C45ModelSelection) modSelection).cleanup();
			((C45ModelSelection) baseModelToForceDecision).cleanup();
		} else {
			super.buildClassifier(instances);
		}
		
		
		
		
	}
	
	/**
	 * Generate as many samples as the number of samples based on Resampling Method parameters concurrently
	 * 
	 * @param instances the training data which will be used to generate the sample set
	 * @param numCore the number of threads going to be used to generate the samples in parallel
	 * @return Instances[] the vector of generated samples
	 * @throws Exception if something goes wrong
	 */
	protected Instances[] generateSamplesParallel(Instances instances, int numCore) throws Exception {
		Instances[] samplesVector = null;
		// can classifier tree handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		InstancesConsolidated instancesWMC = new InstancesConsolidated(instances);
		instancesWMC.deleteWithMissingClass();
		if (m_Debug) {
			System.out.println("=== Generation of the set of samples ===");
			System.out.println(toStringResamplingMethod());
		}
		/** Original sample size */
		int dataSize = instancesWMC.numInstances();
		if(dataSize==0)
			System.err.println("Original data size is 0! Handle zero training instances!");
		else
			if (m_Debug)
				System.out.println("Original data size: " + dataSize);
		/** Size of samples(bags) to be generated */
		int bagSize = 0;

		// Some checks done in set-methods
		//@ requires  0 <= m_RMnumberSamples 
		//@ requires -2 <= m_RMbagSizePercent && m_RMbagSizePercent <= 100 
		//@ requires -2 <= m_RMnewDistrMinClass && m_RMnewDistrMinClass < 100
		if(m_RMbagSizePercent >= 0 ){
			bagSize =  dataSize * m_RMbagSizePercent / 100;
			if(bagSize==0)
				System.err.println("Size of samples is 0 (" + m_RMbagSizePercent + "% of " + dataSize
						+ ")! Handle zero training instances!");
		} else if (m_RMnewDistrMinClass < 0) { // stratified OR free
			throw new Exception("Size of samples, m_RMbagSizePercent, (" + m_RMbagSizePercent + 
					") has to be between 0 and 100, when m_RMnewDistrMinClass < 0 (stratified or free)!!!");
		}

		Random random; 
		if (dataSize == 0) // To be OK when testing to Handle zero training instances!
			random = new Random(m_Seed);
		else
			random = instancesWMC.getRandomNumberGenerator(m_Seed);

		// Generate the vector of samples with the given parameters
		// TODO Set the different options to generate the samples like a filter and then use it here.  
		if(m_RMnewDistrMinClass == (float)-2)
			// stratified: Maintains the original class distribution in the new samples
			samplesVector = generateStratifiedSamplesParallel(instancesWMC, dataSize, bagSize, random, numCore);
		else if (m_RMnewDistrMinClass == (float)-1)
			// free: It doesn't take into account the original class distribution
			samplesVector = generateFreeDistrSamplesParallel(instancesWMC, dataSize, bagSize, random, numCore);
		else
			// RMnewDistrMinClass is between 0 and 100: Changes the class distribution to the indicated value
			samplesVector = generateSamplesChangingMinClassDistrParallel(instancesWMC, dataSize, bagSize, random, numCore);
		if (m_Debug)
			System.out.println("=== End of Generation of the set of samples ===");
		return samplesVector;
	}
	
	/**
	 * Generate a set of stratified samples concurrently
	 * 
	 * @param instances the training data which will be used to generate the sample set
	 * @param dataSize Size of original sample (instances)
	 * @param bagSize Size of samples(bags) to be generated
	 * @param random a random number generator
	 * @param numCore the number of threads going to be used to generate the samples in parallel
	 * @return Instances[] the vector of generated samples
	 * @throws Exception if something goes wrong
	 */
	private Instances[] generateStratifiedSamplesParallel(
			InstancesConsolidated instances, int dataSize, int bagSize, Random random, int numCore) throws Exception{
		int numClasses = instances.numClasses();
		// Get the classes
		InstancesConsolidated[] classesVector =  instances.getClasses();
		// What is the minority class?
		/** Vector containing the size of each class */
		int classSizeVector[] = instances.getClassesSize(classesVector);
		/** Index of the minority class in the original sample */
		int iMinClass = Utils.minIndex(classSizeVector);
		if (m_Debug)
			instances.printClassesInformation(dataSize , iMinClass, classSizeVector);

		// Determine the sizes of each class in the new samples
		/** Vector containing the size of each class in the new samples */
		int newClassSizeVector[] = new int [numClasses];
		// Check the bag size
		int bagSizePercent;
		if((dataSize == bagSize) && !m_RMreplacement){
			System.out.println("It doesn't make sense that the original sample's size and " +
					"the size of samples to be generated are the same without using replacement" +
					"because all the samples will be entirely equal!!!\n" +
					m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
			bagSizePercent = m_bagSizePercentToReduce;
			bagSize =  dataSize * m_bagSizePercentToReduce / 100;
		}
		else
			bagSizePercent = m_RMbagSizePercent;
		/** Partial bag size */
		int localBagSize = 0;
		for(int iClass = 0; iClass < numClasses; iClass++)
			if(iClass != iMinClass){
				/** Value for the 'iClass'-th class size of the samples to be generated */
				int newClassSize = Utils.round(classSizeVector[iClass] * (double)bagSizePercent / 100);
				newClassSizeVector[iClass] = newClassSize;
				localBagSize += newClassSize;
			}
		/** Value for the minority class size of the samples to be generated */
		// (Done in this way to know the exact size of the minority class in the generated samples)
		newClassSizeVector[iMinClass] = bagSize - localBagSize;
		if (m_Debug) {
			System.out.println("New bag size: " + bagSize);
			System.out.println("Classes sizes of the new bag:");
			for (int iClass = 0; iClass < numClasses; iClass++){
				System.out.print(newClassSizeVector[iClass]);
				if(iClass < numClasses - 1)
					System.out.print(", ");
			}
			System.out.println("");
		}
		// Determine the size of samples' vector; the number of samples
		int numberSamples;
		/** Calculate the ratio of the sizes for each class between the sample and the subsample */
		double bagBySampleClassRatioVector[] = new double[numClasses];
		for(int iClass = 0; iClass < numClasses; iClass++)
			if (classSizeVector[iClass] > 0)
				bagBySampleClassRatioVector[iClass] = newClassSizeVector[iClass] / (double)classSizeVector[iClass];
			else // The size of the class is 0
				// This class won't be selected
				bagBySampleClassRatioVector[iClass] = Double.MAX_VALUE;
		if(m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage) {
			// The number of samples depends on the coverage to be guaranteed for the most disfavored class.
			double coverage = m_RMnumberSamples / (double)100;
			/** Calculate the most disfavored class in respect of coverage */
			int iMostDisfavorClass = Utils.minIndex(bagBySampleClassRatioVector);
			if (m_Debug) {
				System.out.println("Ratio bag:sample by each class:");
				System.out.println("(*) The most disfavored class based on coverage");
				for (int iClass = 0; iClass < numClasses; iClass++){
					System.out.print(Utils.doubleToString(bagBySampleClassRatioVector[iClass],2));
					if(iClass == iMostDisfavorClass)
						System.out.print("(*)");
					if(iClass < numClasses - 1)
						System.out.print(", ");
				}
				System.out.println("");
			}
			if(m_RMreplacement)
				numberSamples = (int) Math.ceil((-1) * Math.log(1 - coverage) / 
						bagBySampleClassRatioVector[iMostDisfavorClass]);
			else
				numberSamples = (int) Math.ceil(Math.log(1 - coverage) / 
						Math.log(1 - bagBySampleClassRatioVector[iMostDisfavorClass]));
			System.out.println("The number of samples to guarantee at least a coverage of " +
					Utils.doubleToString(100*coverage,0) + "% is " + numberSamples + ".");
			m_numberSamplesByCoverage = numberSamples;
			if (numberSamples < 3){
				numberSamples = 3;
				System.out.println("(*) Forced the number of samples to be 3!!!");
				m_stExceptionalSituationsMessage += " (*) Forced the number of samples to be 3!!!\n";
			}
		} else // m_RMnumberSamplesHowToSet == NumberSamples_FixedValue 
			// The number of samples has been set by parameter
			numberSamples = (int)m_RMnumberSamples;

		// Calculate the true coverage achieved
		m_trueCoverage = (double)0.0;
		for (int iClass = 0; iClass < numClasses; iClass++){
			double trueCoverageByClass;
			if(classSizeVector[iClass] > 0){
				if(m_RMreplacement)
					trueCoverageByClass = 1 - Math.pow(Math.E, (-1) * bagBySampleClassRatioVector[iClass] * numberSamples);
				else
					trueCoverageByClass = 1 - Math.pow((1 - bagBySampleClassRatioVector[iClass]), numberSamples);
			} else
				trueCoverageByClass = (double)0.0;
			double ratioClassDistr = classSizeVector[iClass] / (double)dataSize;
			m_trueCoverage += ratioClassDistr * trueCoverageByClass;
		}

		// Set the size of the samples' vector 
		Instances[] samplesVector = new Instances[numberSamples];
		
		//ExecutorService klaseko objektu bat sortuko du. Objektuak lan asinkronoak beteko dituen 'thread' edo hariak sortuko ditu.
		
		//newFixedThreadPool(i): Creates a thread pool that reuses a fixed number of threads operating off a shared unbounded queue. 
		//At any point, at most nThreads threads will be active processing tasks.If additional tasks are submitted when all threads
		//are active,they will wait in the queue until a thread is available.If any thread terminates due to a failure during 
		//execution prior to shutdown, a new one will take its place if needed to execute subsequent tasks. The threads in the pool 
		//will exist until it is explicitly shutdown.
		ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
		
		//'Erloju' moduko bat sortzen du non eramango duen kontua sortu beharreko lagin kopurua izango den.
		//Ahalbidetuko du hari batzuk besteei itxarotea dagokien lana guztiek bukatu arte
		final CountDownLatch doneSignal = new CountDownLatch(numberSamples);
		
		//Atomikoki berritzen den Integer balioa. Hari batek ezingo du balio hau berritu jada beste hari bat badago lan berdina egiten
		final AtomicInteger numFailed = new AtomicInteger();
		
		

		// Generate the vector of samples 
		for(int iSample = 0; iSample < numberSamples; iSample++){
			
			final int currentSample = iSample;
			final InstancesConsolidated[] currentClassesVector = classesVector;
			final int[] currentNewClassSizeVector = newClassSizeVector;
			final int[] currentClassSizeVector = classSizeVector;
			
			
			Runnable newTask = new Runnable() {
				public void run() {
					
					try {
						InstancesConsolidated bagData = null;
						InstancesConsolidated bagClass = null;
					
						for(int iClass = 0; iClass < numClasses; iClass++){
							// Extract instances of the iClass-th class
							if(m_RMreplacement)
								bagClass = new InstancesConsolidated(currentClassesVector[iClass].resampleWithWeights(random));
							else
								bagClass = new InstancesConsolidated(currentClassesVector[iClass]);
							// Shuffle the instances
							bagClass.randomize(random);
							if (currentNewClassSizeVector[iClass] < currentClassSizeVector[iClass]) {
								InstancesConsolidated newBagData = new InstancesConsolidated(bagClass, 0, currentNewClassSizeVector[iClass]);
								bagClass = newBagData;
								newBagData = null;
							}
							if(bagData == null)
								bagData = bagClass;
							else
								bagData.add(bagClass);
							bagClass = null;
						}
		
						// Shuffle the instances
						bagData.randomize(random);
						samplesVector[currentSample] = (Instances)bagData;
						bagData = null;
						
					} catch (Throwable var) {
						var.printStackTrace();
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
		classesVector = null;
		classSizeVector = null;
		newClassSizeVector = null;
		executorPool.shutdownNow();

		return samplesVector;
	}
	
	/**
	 * Generate a set of samples concurrently without taking the class distribution into account
	 * (like in the meta-classifier Bagging)
	 * 
	 * @param instances the training data which will be used to generate the sample set
	 * @param dataSize Size of original sample (instances)
	 * @param bagSize Size of samples(bags) to be generated
	 * @param random a random number generator
	 * @param numCore the number of threads going to be used to generate the samples in parallel
	 * @return Instances[] the vector of generated samples
	 * @throws Exception if something goes wrong
	 */
	private Instances[] generateFreeDistrSamplesParallel(
			InstancesConsolidated instances, int dataSize, int bagSize, Random random, int numCore) throws Exception{
		// Check the bag size
		if((dataSize == bagSize) && !m_RMreplacement){
			System.out.println("It doesn't make sense that the original sample's size and " +
					"the size of samples to be generated are the same without using replacement" +
					"because all the samples will be entirely equal!!!\n" +
					m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
			bagSize =  dataSize * m_bagSizePercentToReduce / 100;
		}
		if (m_Debug)
			System.out.println("New bag size: " + bagSize);
		// Determine the size of samples' vector; the number of samples
		int numberSamples;
		double bagBySampleRatio = bagSize / (double) dataSize;
		if(m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage) {
			// The number of samples depends on the coverage to be guaranteed for the most disfavored class.
			double coverage = m_RMnumberSamples / (double)100;
			if(m_RMreplacement)
				numberSamples = (int) Math.ceil((-1) * Math.log(1 - coverage) / 
						bagBySampleRatio);
			else
				numberSamples = (int) Math.ceil(Math.log(1 - coverage) / 
						Math.log(1 - bagBySampleRatio));
			System.out.println("The number of samples to guarantee at least a coverage of " +
					Utils.doubleToString(100*coverage,0) + "% is " + numberSamples + ".");
			m_numberSamplesByCoverage = numberSamples;
			if (numberSamples < 3){
				numberSamples = 3;
				System.out.println("(*) Forced the number of samples to be 3!!!");
				m_stExceptionalSituationsMessage += " (*) Forced the number of samples to be 3!!!\n";
			}
		} else // m_RMnumberSamplesHowToSet == NumberSamples_FixedValue
			// The number of samples has been set by parameter
			numberSamples = (int)m_RMnumberSamples;

		// Calculate the true coverage achieved
		if(m_RMreplacement)
			m_trueCoverage = 1 - Math.pow(Math.E, (-1) * bagBySampleRatio * numberSamples);
		else
			m_trueCoverage = 1 - Math.pow((1 - bagBySampleRatio), numberSamples);

		// Set the size of the samples' vector 
		Instances[] samplesVector = new Instances[numberSamples];
		
		//ExecutorService klaseko objektu bat sortuko du. Objektuak lan asinkronoak beteko dituen 'thread' edo hariak sortuko ditu.
		
		//newFixedThreadPool(i): Creates a thread pool that reuses a fixed number of threads operating off a shared unbounded queue. 
		//At any point, at most nThreads threads will be active processing tasks.If additional tasks are submitted when all threads
		//are active,they will wait in the queue until a thread is available.If any thread terminates due to a failure during 
		//execution prior to shutdown, a new one will take its place if needed to execute subsequent tasks. The threads in the pool 
		//will exist until it is explicitly shutdown.
		ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
				
		//'Erloju' moduko bat sortzen du non eramango duen kontua sortu beharreko lagin kopurua izango den.
		//Ahalbidetuko du hari batzuk besteei itxarotea dagokien lana guztiek bukatu arte
		final CountDownLatch doneSignal = new CountDownLatch(numberSamples);
				
		//Atomikoki berritzen den Integer balioa. Hari batek ezingo du balio hau berritu jada beste hari bat badago lan berdina egiten
		final AtomicInteger numFailed = new AtomicInteger();

		// Generate the vector of samples 
		for(int iSample = 0; iSample < numberSamples; iSample++){
			
			final int currentSample = iSample;
			final int currentBagSize = bagSize;
			
			Runnable newTask = new Runnable() {
				public void run() {
					try {
						Instances bagData = null;
						if(m_RMreplacement)
							bagData = new Instances(instances.resampleWithWeights(random));
						else
							bagData = new Instances(instances);
						// Shuffle the instances
						bagData.randomize(random);
						if (currentBagSize < dataSize) {
							Instances newBagData = new Instances(bagData, 0, currentBagSize);
							bagData = newBagData;
							newBagData = null;
						}
						samplesVector[currentSample] = bagData;
						bagData = null;
					} catch (Throwable var) {
						var.printStackTrace();
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
		
		return samplesVector;
	}
	
	/**
	 * Generate a set of samples changing the distribution of the minority class concurrently
	 * 
	 * @param instances the training data which will be used to generate the sample set
	 * @param dataSize Size of original sample (instances)
	 * @param bagSize Size of samples(bags) to be generated
	 * @param random a random number generator
	 * @param numCore the number of threads going to be used to generate the samples in parallel
	 * @return Instances[] the vector of generated samples
	 * @throws Exception if something goes wrong
	 */
	private Instances[] generateSamplesChangingMinClassDistrParallel(
			InstancesConsolidated instances, int dataSize, int bagSize, Random random, int numCore) throws Exception{
		int numClasses = instances.numClasses();
		// Some checks
		if((numClasses > 2) && (m_RMnewDistrMinClass != (float)50.0))
			throw new Exception("In the case of multi-class datasets, the only posibility to change the distribution of classes is to balance them!!!\n" +
					"Use the special value '50.0' in <distribution minority class> for this purpose!!!");
		// TODO Generalize the process to multi-class datasets to set different new values of distribution for each classs.
		// Some checks done in set-methods
		//@ requires m_RMreplacement = false 
		// TODO Accept replacement

		// Get the classes
		InstancesConsolidated[] classesVector =  instances.getClasses();

		// What is the minority class?
		/** Vector containing the size of each class */
		int classSizeVector[] = instances.getClassesSize(classesVector);
		/** Index of the minority class in the original sample */
		int iMinClass, i_iMinClass;
		/** Prevent the minority class from being empty (we hope there is one non-empty!) */
		int iClassSizeOrdVector[] = Utils.sort(classSizeVector);
		for(i_iMinClass = 0; ((i_iMinClass < numClasses) && (classSizeVector[iClassSizeOrdVector[i_iMinClass]] == 0)); i_iMinClass++);
		if(i_iMinClass < numClasses)
			iMinClass = iClassSizeOrdVector[i_iMinClass];
		else // To be OK when testing to Handle zero training instances!
			iMinClass = 0;

		/** Index of the majority class in the original sample */
		int iMajClass = Utils.maxIndex(classSizeVector);
		/** Determines whether the original sample is balanced or not */
		boolean isBalanced = false;
		if (iMinClass == iMajClass){
			isBalanced = true;
			// If the sample is balanced, it is determined, by convention, that the majority class is the last one
			iMajClass = numClasses-1;
		}
		if (m_Debug)
			instances.printClassesInformation(dataSize , iMinClass, classSizeVector);

		/** Distribution of the minority class in the original sample */
		float distrMinClass;
		if (dataSize == 0)
			distrMinClass = (float)0;
		else
			distrMinClass = (float)100 * classSizeVector[iMinClass] / dataSize;

		/** Guarantee the minimum number of examples in each class based on m_minExamplesPerClassPercent */
		int minExamplesPerClass = (int) Math.ceil(dataSize * m_minExamplesPerClassPercent / (double)100.0) ;
		/** Guarantee to be at least m_minNumObj */
		if (minExamplesPerClass < m_minNumObj)
			minExamplesPerClass = m_minNumObj;
		if (m_Debug)
			System.out.println("Minimum number of examples to be guaranteed in each class: " + minExamplesPerClass);
		for(int iClass = 0; iClass < numClasses; iClass++){
			if((classSizeVector[iClass] < minExamplesPerClass) && // if number of examples is smaller than the minimum
					(classSizeVector[iClass] > 0)){					// but, at least, it has to exist any example.
				// Oversample the class ramdonly
				System.out.println("The " + iClass + "-th class has too few examples (" + classSizeVector[iClass]+ ")!!!\n" +
						"It will be oversampled ranmdoly up to " + minExamplesPerClass + "!!!");
				m_stExceptionalSituationsMessage += " (*) Forced the " + iClass + "-th class to be oversampled!!!\n";
				// based on the code of the function 'resample(Random)' of the class 'Instances'
				InstancesConsolidated bagClass = classesVector[iClass];
				while (bagClass.numInstances() < minExamplesPerClass) {
					bagClass.add(classesVector[iClass].instance(random.nextInt(classSizeVector[iClass])));
				}
				// Update the vectors with classes' information and the new data size
				dataSize = dataSize - classSizeVector[iClass] + minExamplesPerClass; 
				classesVector[iClass] = bagClass;
				classSizeVector[iClass] = minExamplesPerClass;
			}
		}

		/** Maximum values for classes' size on the samples to be generated taking RMnewDistrMinClass into account
		 *   and without using replacement */
		int maxClassSizeVector[] = new int[numClasses];
		if (numClasses == 2){
			// the dataset is two-class
			if(m_RMnewDistrMinClass > distrMinClass){
				// Maintains the whole minority class
				maxClassSizeVector[iMinClass] = classSizeVector[iMinClass];
				maxClassSizeVector[iMajClass] = Utils.round(classSizeVector[iMinClass] * (100 - m_RMnewDistrMinClass) / m_RMnewDistrMinClass);
			} else {
				// Maintains the whole majority class
				maxClassSizeVector[iMajClass] = classSizeVector[iMajClass];
				maxClassSizeVector[iMinClass] = Utils.round(classSizeVector[iMajClass] * m_RMnewDistrMinClass / (100 - m_RMnewDistrMinClass));
			}
		} else {
			// the dataset is multi-class
			/** The only accepted option is to change the class distribution is to balance the samples */
			for(int iClass = 0; iClass < numClasses; iClass++)
				maxClassSizeVector[iClass] = classSizeVector[iMinClass];
		}

		// Determine the sizes of each class in the new samples
		/** Vector containing the size of each class in the new samples */
		int newClassSizeVector[] = new int[numClasses];
		/** Determines whether the size of samples to be generated will be forced to be reduced in exceptional situations */
		boolean forceToReduceSamplesSize = false;
		if(m_RMbagSizePercent == -2){
			// maxSize : Generate the biggest samples according to the indicated distribution (RMnewDistrMinClass),
			//  that is, maintaining the whole minority (majority) class
			if (numClasses == 2){
				// the dataset is two-class
				if(Utils.eq(m_RMnewDistrMinClass, distrMinClass)){
					System.out.println("It doesn't make sense that the original distribution and " +
							"the distribution to be changed (RMnewDistrMinClass) are the same and " +
							"the size of samples to be generated is maximum (RMbagSizePercent=-2) " +
							"(without using replacement) " +
							"because all the samples will be entirely equal!!!\n" +
							m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
					forceToReduceSamplesSize = true;
				}
			} else
				// the dataset is multi-class
				if (isBalanced){
					System.out.println("In the case of multi-class datasets, if the original sample is balanced, " +
							"it doesn't make sense that " +
							"the size of samples to be generated is maximum (RMbagSizePercent=-2) " +
							"(without using replacement) " +
							"because all the samples will be entirely equal!!!\n" +
							m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
					forceToReduceSamplesSize = true;
				}
			if(!forceToReduceSamplesSize){
				bagSize = 0;
				for(int iClass = 0; iClass < numClasses; iClass++)
					if (classSizeVector[iClass] == 0)
						newClassSizeVector[iClass] = 0;
					else {
						newClassSizeVector[iClass] = maxClassSizeVector[iClass];
						bagSize += maxClassSizeVector[iClass];
					}
			}
		} else {
			if (m_RMbagSizePercent == -1)
				// sizeOfMinClass: the generated samples will have the same size that the minority class
				bagSize = classSizeVector[iMinClass];
			else
				// m_RMbagSizePercent is between 0 and 100. bagSize is already set.
				if(dataSize == bagSize){
					System.out.println("It doesn't make sense that the original sample's size and " +
							"the size of samples to be generated are the same" +
							"(without using replacement) " +
							"because all the samples will be entirely equal!!!\n" +
							m_bagSizePercentToReduce + "% will be used as the bag size percentage!!!");
					forceToReduceSamplesSize = true;
				}
		}
		if((m_RMbagSizePercent != -2) || forceToReduceSamplesSize)
		{
			if (numClasses == 2){
				// the dataset is two-class
				if(forceToReduceSamplesSize){
					bagSize =  dataSize * m_bagSizePercentToReduce / 100;
					m_stExceptionalSituationsMessage += " (*) Forced to reduce the size of the generated samples!!!\n";
				}
				newClassSizeVector[iMinClass] = Utils.round(m_RMnewDistrMinClass * bagSize / 100);
				newClassSizeVector[iMajClass] = bagSize - newClassSizeVector[iMinClass];
			} else {
				// the dataset is multi-class
				/** The only accepted option is to change the class distribution is to balance the samples */
				/** All the classes will have the same size, classSize, based on bagSizePercent applied 
				 *  on minority class, that is, the generated samples will be bigger than expected, 
				 *  neither sizeofMinClass nor original sample's size by bagSizePercent. Otherwise,
				 *  the classes of the samples would be too unpopulated */
				int bagSizePercent;
				if (m_RMbagSizePercent == -1)
					/** sizeOfMinClass (-1) is a special case, where the bagSizePercent to be applied will be
					 *  the half of the minority class, the same that it would be achieved if the dataset was two-class */
					bagSizePercent = 50;
				else
					if (forceToReduceSamplesSize)
						bagSizePercent = m_bagSizePercentToReduce;
					else
						bagSizePercent = m_RMbagSizePercent;
				int classSize = (int)(bagSizePercent * classSizeVector[iMinClass] / (float)100);
				bagSize = 0;
				for(int iClass = 0; iClass < numClasses; iClass++)
					if (classSizeVector[iClass] == 0)
						newClassSizeVector[iClass] = 0;
					else {
						newClassSizeVector[iClass] = classSize;
						bagSize += classSize;
					}
			}
		}

		if (m_Debug) {
			System.out.println("New bag size: " + bagSize);
			System.out.println("New minority class size: " + newClassSizeVector[iMinClass] + " (" + (int)(newClassSizeVector[iMinClass] / (double)bagSize * 100) + "%)");
			System.out.print("New majority class size: " + newClassSizeVector[iMajClass]);
			if (numClasses > 2)
				System.out.print(" (" + (int)(newClassSizeVector[iMajClass] / (double)bagSize * 100) + "%)");
			System.out.println();
		}
		// Some checks
		for(int iClass = 0; iClass < numClasses; iClass++)
			if(newClassSizeVector[iClass] > classSizeVector[iClass])
				throw new Exception("There aren't enough instances of the " + iClass + 
						"-th class (" +	classSizeVector[iClass] +
						") to extract " + newClassSizeVector[iClass] +
						" for the new samples whithout replacement!!!");

		// Determine the size of samples' vector; the number of samples
		int numberSamples;
		/** Calculate the ratio of the sizes for each class between the sample and the subsample */
		double bagBySampleClassRatioVector[] = new double[numClasses];
		for(int iClass = 0; iClass < numClasses; iClass++)
			if (classSizeVector[iClass] > 0)
				bagBySampleClassRatioVector[iClass] = newClassSizeVector[iClass] / (double)classSizeVector[iClass];
			else // The size of the class is 0
				// This class won't be selected
				bagBySampleClassRatioVector[iClass] = Double.MAX_VALUE;
		if(m_RMnumberSamplesHowToSet == NumberSamples_BasedOnCoverage) {
			// The number of samples depends on the coverage to be guaranteed for the most disfavored class.
			double coverage = m_RMnumberSamples / (double)100;
			/** Calculate the most disfavored class in respect of coverage */
			int iMostDisfavorClass = Utils.minIndex(bagBySampleClassRatioVector);
			if (m_Debug) {
				System.out.println("Ratio bag:sample by each class:");
				System.out.println("(*) The most disfavored class based on coverage");
				for (int iClass = 0; iClass < numClasses; iClass++){
					System.out.print(Utils.doubleToString(bagBySampleClassRatioVector[iClass],2));
					if(iClass == iMostDisfavorClass)
						System.out.print("(*)");
					if(iClass < numClasses - 1)
						System.out.print(", ");
				}
				System.out.println("");
			}
			numberSamples = (int) Math.ceil(Math.log(1 - coverage) / 
					Math.log(1 - bagBySampleClassRatioVector[iMostDisfavorClass]));
			System.out.println("The number of samples to guarantee at least a coverage of " +
					Utils.doubleToString(100*coverage,0) + "% is " + numberSamples + ".");
			m_numberSamplesByCoverage = numberSamples;
			if (numberSamples < 3){
				numberSamples = 3;
				System.out.println("(*) Forced the number of samples to be 3!!!");
				m_stExceptionalSituationsMessage += " (*) Forced the number of samples to be 3!!!\n";
			}
		} else // m_RMnumberSamplesHowToSet == NumberSamples_FixedValue
			// The number of samples has been set by parameter
			numberSamples = (int)m_RMnumberSamples;

		// Calculate the true coverage achieved
		m_trueCoverage = (double)0.0;
		for (int iClass = 0; iClass < numClasses; iClass++){
			double trueCoverageByClass;
			if(classSizeVector[iClass] > 0){
				if(m_RMreplacement)
					trueCoverageByClass = 1 - Math.pow(Math.E, (-1) * bagBySampleClassRatioVector[iClass] * numberSamples);
				else
					trueCoverageByClass = 1 - Math.pow((1 - bagBySampleClassRatioVector[iClass]), numberSamples);
			} else
				trueCoverageByClass = (double)0.0;
			double ratioClassDistr = classSizeVector[iClass] / (double)dataSize;
			m_trueCoverage += ratioClassDistr * trueCoverageByClass;
		}

		// Set the size of the samples' vector 
		Instances[] samplesVector = new Instances[numberSamples];
		
		//ExecutorService klaseko objektu bat sortuko du. Objektuak lan asinkronoak beteko dituen 'thread' edo hariak sortuko ditu.
		
		//newFixedThreadPool(i): Creates a thread pool that reuses a fixed number of threads operating off a shared unbounded queue. 
		//At any point, at most nThreads threads will be active processing tasks.If additional tasks are submitted when all threads
		//are active,they will wait in the queue until a thread is available.If any thread terminates due to a failure during 
		//execution prior to shutdown, a new one will take its place if needed to execute subsequent tasks. The threads in the pool 
		//will exist until it is explicitly shutdown.
		ExecutorService executorPool = Executors.newFixedThreadPool(numCore);
		
		//'Erloju' moduko bat sortzen du non eramango duen kontua klasifikatzaileen kopurua izango den.
		//Ahalbidetuko du hari batzuk besteei itxarotea dagokien lana guztiek bukatu arte
		final CountDownLatch doneSignal = new CountDownLatch(numberSamples);
		
		//Atomikoki berritzen den Integer balioa. Hari batek ezingo du balio hau berritu jada beste hari bat badago lan berdina egiten
		final AtomicInteger numFailed = new AtomicInteger();
		
		final InstancesConsolidated[] currentClassesVector = classesVector;
		final int[] currentNewClassSizeVector = newClassSizeVector;
		final int[] currentClassSizeVector = classSizeVector;

		// Generate the vector of samples 
		for(int iSample = 0; iSample < numberSamples; iSample++){
			
			final int currentSample = iSample;
			
			Runnable newTask = new Runnable() {
				public void run() {
					try {
						InstancesConsolidated bagData = null;
						InstancesConsolidated bagClass = null;
						
						for (int iClass = 0; iClass < numClasses; iClass++)
							if (currentClassSizeVector[iClass] > 0){
								// Extract instances of the i-th class
								bagClass = new InstancesConsolidated(currentClassesVector[iClass]);
								// Shuffle the instances
								bagClass.randomize(random);
								if (currentNewClassSizeVector[iClass] < currentClassSizeVector[iClass]) {
									InstancesConsolidated newBagData = new InstancesConsolidated(bagClass, 0, currentNewClassSizeVector[iClass]);
									bagClass = newBagData;
									newBagData = null;
								}
								// Add the bagClass (i-th class) to bagData
								if (bagData == null)
									bagData = bagClass;
								else
									bagData.add(bagClass);
								bagClass = null;
							}
						// Shuffle the instances
						if (bagData == null) // To be OK when testing to Handle zero training instances!
							bagData = instances;
						else
							bagData.randomize(random);
						samplesVector[currentSample] = (Instances)bagData;
						bagData = null;
						
					} catch (Throwable var) {
						var.printStackTrace();
						numFailed.incrementAndGet();
						System.out.println("Iteration " + currentSample + " failed!");
					} finally {
						doneSignal.countDown();
					}
					
				}
			};
			executorPool.submit(newTask);
			
		}
		doneSignal.await();
		executorPool.shutdownNow();
		
		classesVector = null;
		classSizeVector = null;
		maxClassSizeVector = null;
		newClassSizeVector = null;
		
		return samplesVector;
	}
	
	
	
	
	
	
	
}
