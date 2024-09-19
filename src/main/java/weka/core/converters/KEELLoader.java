/**
 * 
 */
package weka.core.converters;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StreamTokenizer;
import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * <!-- globalinfo-start -->
 * Reads a source that is in KEEL (Knowledge Extraction based on Evolutionary Learning) format.
 * http://www.keel.es/
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @version $Revision: 0 $
 * @see Loader
*/
public class KEELLoader extends AbstractFileLoader  implements BatchConverter {

	/** for serialization */
	private static final long serialVersionUID = 2066758300434334831L;

	/** the file extension */
	public static String FILE_EXTENSION = ".dat";

	/** The keyword used to denote the start of a list of the attributes which will be processed as inputs.*/
	public final static String KEEL_INPUTS = "@inputs";

	/** The keyword used to denote the start of a list of the attributes which will be processed as outputs.*/
	public final static String KEEL_OUTPUTS = "@outputs";

	/** The keyword used to denote a missing value */
	public final static String KEEL_MISSING_VALUE = "<null>";

	/** The reader for the source file. */
	protected transient Reader m_sourceReader = null;

    /** the tokenizer for reading the stream */
    protected StreamTokenizer m_Tokenizer;

    /**
	 * Which attributes are ignore, that is, are not part of the list of inputs
	 * (based on C45Loader class)
	 */
	protected boolean[] m_ignore;

	/**
	 * Get the file extension used for arff files
	 * 
	 * @return the file extension
	 */
	@Override
	public String getFileExtension() {
	    return FILE_EXTENSION;
	}

	/**
	 * Gets all the file extensions used for this type of file
	 * 
	 * @return the file extensions
	 */
	@Override
	public String[] getFileExtensions() {
	    return new String[] { FILE_EXTENSION, ".keel" };
	}

	/**
	 * Returns a description of the file type.
	 * 
	 * @return a short file description
	 */
	@Override
	public String getFileDescription() {
		return "KEEL data files";
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 0 $");
	}

	/**
	 * Resets the Loader object and sets the source of the data set to be 
	 * the supplied InputStream.
	 *
	 * @param in 			the source InputStream.
	 * @throws IOException 	if initialization of reader fails.
	 */
	public void setSource(InputStream in) throws IOException {
		m_File = (new File(System.getProperty("user.dir"))).getAbsolutePath();

		m_sourceReader = new BufferedReader(new InputStreamReader(in));
	}

	/**
	 * Initializes the stream tokenizer
	 */
	private void initTokenizer() {
		m_Tokenizer.resetSyntax();
		//m_Tokenizer.whitespaceChars(0, (' ' - 1));
		m_Tokenizer.whitespaceChars(0, ' ');
		//m_Tokenizer.wordChars(' ', '\u00FF');
		m_Tokenizer.wordChars(' ' + 1, '\u00FF');
		m_Tokenizer.whitespaceChars(',', ',');
		m_Tokenizer.whitespaceChars(':', ':');
		// m_Tokenizer.whitespaceChars('.','.');
		m_Tokenizer.commentChar('|');
		m_Tokenizer.commentChar('%');
		m_Tokenizer.commentChar('['); // In order to ignore min an max values
		m_Tokenizer.commentChar(']');
		m_Tokenizer.whitespaceChars('\t', '\t');
		m_Tokenizer.quoteChar('"');
		m_Tokenizer.quoteChar('\'');
		m_Tokenizer.ordinaryChar('{');
		m_Tokenizer.ordinaryChar('}');
//		m_Tokenizer.ordinaryChar('<');
//		m_Tokenizer.ordinaryChar('>');
		m_Tokenizer.eolIsSignificant(true);
	}

	/**
	 * Gets token, skipping empty lines.
	 * (based on StreamTokenizerUtils class)
	 * 
	 * @throws IOException if reading the next token fails
	 */
	public void getFirstToken()
			throws IOException {

		while (m_Tokenizer.nextToken() == StreamTokenizer.TT_EOL) {
		}
		;
		if ((m_Tokenizer.ttype == '\'') || (m_Tokenizer.ttype == '"')) {
			m_Tokenizer.ttype = StreamTokenizer.TT_WORD;
		} else if ((m_Tokenizer.ttype == StreamTokenizer.TT_WORD)
				&& (m_Tokenizer.sval.equals("?") || m_Tokenizer.sval.equals(KEEL_MISSING_VALUE))
				) {
			m_Tokenizer.ttype = '?';
		}
	}

	/**
	 * Gets token.
	 * (based on StreamTokenizerUtils class)
	 * 
	 * @throws IOException if reading the next token fails
	 */
	public void getToken() throws IOException {

		m_Tokenizer.nextToken();
		if (m_Tokenizer.ttype == StreamTokenizer.TT_EOL) {
			return;
		}

		if ((m_Tokenizer.ttype == '\'') || (m_Tokenizer.ttype == '"')) {
			m_Tokenizer.ttype = StreamTokenizer.TT_WORD;
		} else if ((m_Tokenizer.ttype == StreamTokenizer.TT_WORD)
				&& (m_Tokenizer.sval.equals("?") || m_Tokenizer.sval.equals(KEEL_MISSING_VALUE)) 
				) {
			m_Tokenizer.ttype = '?';
		}
	}

	/**
	 * Throws error message with line number and last token read.
	 * (based on StreamTokenizerUtils class)
	 * 
	 * @param theMsg the error message to be thrown
	 * @throws IOException containing the error message
	 */
	public void errms(String theMsg)
			throws IOException {

		throw new IOException(theMsg + ", read " + m_Tokenizer.toString());
	}

	/**
	 * Gets token and checks if its end of line.
	 * (based on ArffLoader class)
	 * 
	 * @param endOfFileOk whether EOF is OK
	 * @throws IOException if it doesn't find an end of line
	 */
	protected void getLastToken(boolean endOfFileOk) throws IOException {
		if ((m_Tokenizer.nextToken() != StreamTokenizer.TT_EOL)
				&& ((m_Tokenizer.ttype != StreamTokenizer.TT_EOF) || !endOfFileOk)) {
			errms("end of line expected");
		}
	}

	/**
	 * Reads and skips all tokens before next end of line token.
	 * (based on ArffLoader class)
	 * 
	 * @throws IOException in case something goes wrong
	 */
	protected void readTillEOL() throws IOException {
		while (m_Tokenizer.nextToken() != StreamTokenizer.TT_EOL) {
		}

		m_Tokenizer.pushBack();
	}

	/**
	 * Parses the attribute declaration.
	 * (based on ArffLoader class)
	 * 
	 * @param attributes the current attributes vector
	 * @return the new attributes vector
	 * @throws IOException if the information is not read successfully
	 */
	protected ArrayList<Attribute> parseAttribute(
			ArrayList<Attribute> attributes) throws IOException {
		String attributeName;
		ArrayList<String> attributeValues;

		// Get attribute name.
		getToken();
		attributeName = m_Tokenizer.sval;
		getToken();

		// Check if attribute is nominal.
		if (m_Tokenizer.ttype == StreamTokenizer.TT_WORD) {

			// Attribute is real, integer, or string.
			if (m_Tokenizer.sval.equalsIgnoreCase(Attribute.ARFF_ATTRIBUTE_REAL)
					|| m_Tokenizer.sval
					.equalsIgnoreCase(Attribute.ARFF_ATTRIBUTE_INTEGER)
					// ARFF_ATTRIBUTE_NUMERIC is not considered in KEEL
					) {
				Attribute att = new Attribute(attributeName, attributes.size());
				att.setWeight(1.0);
				attributes.add(att);
				readTillEOL();
			} else
				// ARFF_ATTRIBUTE_STRING is not considered in KEEL
				// ARFF_ATTRIBUTE_DATE is not considered in KEEL
				// ARFF_ATTRIBUTE_RELATIONAL is not considered in KEEL
			{
				errms("no valid attribute type or invalid " + "enumeration");
			}
		} else {

			// Attribute is nominal.
			attributeValues = new ArrayList<String>();
			m_Tokenizer.pushBack();

			// Get values for nominal attribute.
			if (m_Tokenizer.nextToken() != '{') {
				errms("{ expected at beginning of enumeration");
			}
			while (m_Tokenizer.nextToken() != '}') {
				if (m_Tokenizer.ttype == StreamTokenizer.TT_EOL) {
					errms("} expected at end of enumeration");
				} else {
					attributeValues.add(m_Tokenizer.sval);
				}
			}
			Attribute att = new Attribute(attributeName, attributeValues, attributes.size());
			att.setWeight(1.0);
			attributes.add(att);
			readTillEOL();
		}
		getLastToken(false);
		getFirstToken();
		if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
			errms("premature end of file");
		}

		return attributes;
	}

	/**
	 * Reads and stores header of an KEEL file.
	 * (based on ArffLoader class)
	 * 
	 * @return the structure of the data set as an empty set of Instances
	 * @throws IOException if the information is not read successfully
	 */
	protected Instances readHeader() throws IOException {

		String relationName = "";
		int count, i_class;

		// Get name of relation.
		getFirstToken();
		if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
			errms("premature end of file");
		}
		if (Instances.ARFF_RELATION.equalsIgnoreCase(m_Tokenizer.sval)) {
			getToken();
			relationName = m_Tokenizer.sval;
			getLastToken(false);
		} else {
			errms("keyword " + Instances.ARFF_RELATION + " expected");
		}

		// Create vectors to hold information temporarily.
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		// Get attribute declarations.
		getFirstToken();
		if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
			errms("premature end of file");
		}

		while (Attribute.ARFF_ATTRIBUTE.equalsIgnoreCase(m_Tokenizer.sval)) {
			attributes = parseAttribute(attributes);
		}

		// Check if any attributes have been declared.
		if (attributes.size() == 0) {
			errms("no attributes declared");
		}

		Instances data = new Instances(relationName, attributes, 1000);

		// Get input attributes vector.
		m_ignore = new boolean[attributes.size()];
		if (KEEL_INPUTS.equalsIgnoreCase(m_Tokenizer.sval)) {
			count = 0;
			Arrays.fill(m_ignore, true);
			while (m_Tokenizer.nextToken() != StreamTokenizer.TT_EOL) {
				Attribute att = data.attribute(m_Tokenizer.sval);
				if (att != null) {
					m_ignore[att.index()] = false;
					count++;
				}
				else
					errms("inputs: '" + m_Tokenizer.sval + "' attribute was not defined");
			}
			//readTillEOL();
			if (count == 0) {
				errms("No inputs defined");
			}
			// Get output attributes vector.
			getFirstToken();
			if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
				errms("premature end of file");
			}
		} else {
			// @inputs doesn't apper. All attributes will be set as valid inputs
			//errms("keyword " + KEEL_INPUTS + " expected");
			Arrays.fill(m_ignore, false);
		}
		if (KEEL_OUTPUTS.equalsIgnoreCase(m_Tokenizer.sval)) {
			count = 0;
			i_class = -1;
			while ((m_Tokenizer.nextToken() != StreamTokenizer.TT_EOL) &&
					(count < 2)) {
				Attribute att = data.attribute(m_Tokenizer.sval);
				if (att != null) {
					i_class = att.index();
					count++;
				}
				else
					errms("outputs: '" + m_Tokenizer.sval + "' attribute was not defined");
			}
			//readTillEOL();
			if (count == 0) {
				errms("No outputs defined");
			} else if (count == 1) {
				data.setClassIndex(i_class);
				m_ignore[i_class] = false;
			} else {
				errms("Only one output can be set as class");
			}
			// Check if data part follows. We can't easily check for EOL.
			getFirstToken();
			if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
				errms("premature end of file");
			}
		} else {
			// @outputs doesn't apper. The last attribute will be set as the class
			//errms("keyword " + KEEL_OUTPUTS + " expected");
			i_class = data.numAttributes() - 1;
			data.setClassIndex(i_class);
			m_ignore[i_class] = false;
		}

		if (!Instances.ARFF_DATA.equalsIgnoreCase(m_Tokenizer.sval)) {
			errms("keyword " + Instances.ARFF_DATA + " expected");
		}
		return data;
	}

	/**
	 * Determines and returns (if possible) the structure (internally the header)
	 * of the data set as an empty set of instances.
	 * 
	 * @return the structure of the data set as an empty set of Instances
	 * @exception IOException if an error occurs
	 */
	@Override
	public Instances getStructure() throws IOException {

		if (m_sourceReader == null) {
			throw new IOException("No source has been specified");
		}

		if (m_structure == null) {
			m_Tokenizer = new StreamTokenizer(m_sourceReader);
			initTokenizer();
			m_structure = readHeader();
		}

		return m_structure;
	}

	/**
	 * Reads a single instance using the tokenizer and returns it.
	 * 
	 * @param structure the dataset header information
	 * @return null if end of file has been reached
	 * @throws IOException if the information is not read successfully
	 */
	protected Instance getInstanceFull(Instances structure) throws IOException {
		double[] instance = new double[structure.numAttributes()];
		int index;

		// Get values for all attributes.
		for (int i = 0; i < structure.numAttributes(); i++) {
			// Get next token
			if (i > 0) {
				getToken();
			}
			if (!m_ignore[i]) {

				// Check if value is missing.
				if (m_Tokenizer.ttype == '?') {
					instance[i] = Utils.missingValue();
				} else {
					// Check if token is valid.
					if (m_Tokenizer.ttype != StreamTokenizer.TT_WORD) {
						errms("not a valid value");
					}
					switch (structure.attribute(i).type()) {
					case Attribute.NOMINAL:
						// Check if value appears in header.
						index = structure.attribute(i).indexOfValue(m_Tokenizer.sval);
						if (index == -1) {
							errms("nominal value not declared in header");
						}
						instance[i] = index;
						break;
					case Attribute.NUMERIC:
						// Check if value is really a number.
						try {
							instance[i] = Double.valueOf(m_Tokenizer.sval).doubleValue();
						} catch (NumberFormatException e) {
							errms("number expected");
						}
						break;
//					case Attribute.STRING:
//					case Attribute.DATE:
//					case Attribute.RELATIONAL:
					default:
						errms("unknown attribute type in column " + i);
					}
				}
			}
		}

		// Add instance to dataset
		Instance inst = new DenseInstance(1.0, instance);
		inst.setDataset(structure);

		return inst;
	}

	/**
	 * Reads a single instance using the tokenizer and returns it.
	 * 
	 * @param structure the dataset header information
	 * @return null if end of file has been reached
	 * @throws IOException if the information is not read successfully
	 */
	@Override
	public Instance getNextInstance(Instances structure) throws IOException {
		// Check if any attributes have been declared.
		if (structure.numAttributes() == 0) {
			errms("no header information available");
		}

		// Check if end of file reached.
		getFirstToken();
		if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
			return null;
		}

		// Parse instance
		return getInstanceFull(structure);
	}

	/**
	 * Return the full data set.
	 * (based on ArffLoader class)
	 * 
	 * @return the structure of the data set as an empty set of Instances
	 * @throws IOException if there is no source or parsing fails
	 */
	@Override
	public Instances getDataSet() throws IOException {
		Instances insts = null;
		try {
			if (m_sourceReader == null) {
				throw new IOException("No source has been specified");
			}
			if (getRetrieval() == INCREMENTAL) {
				throw new IOException(
						"Cannot mix getting Instances in both incremental and batch modes");
			}
			setRetrieval(BATCH);
			if (m_structure == null) {
				getStructure();
			}

			// Read all instances
			insts = new Instances(m_structure, 0);
			Instance inst;
			while ((inst = getNextInstance(m_structure)) != null) {
				insts.add(inst);
			}
		} finally {
			if (m_sourceReader != null) {
				// close the stream
				m_sourceReader.close();
			}
		}

		return insts;
	}
}
