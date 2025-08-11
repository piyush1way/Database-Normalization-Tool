import streamlit as st
import pandas as pd
import io
import base64
from typing import Dict, List, Set, Tuple 
import re
import itertools
from collections import defaultdict
import graphviz
import zipfile

#  Streamlit
st.set_page_config(
    page_title="Database Normalization Tool",
    page_icon="🗃️",
    layout="wide"
)

class FunctionDependency:
    """Represents a functional dependency A -> B"""
    
    def __init__(self, determinant: Set[str], dependent: Set[str]):
        self.determinant = frozenset(determinant)
        self.dependent = frozenset(dependent)
    
    def __str__(self):
        det_str = ', '.join(sorted(self.determinant))
        dep_str = ', '.join(sorted(self.dependent))
        return f"{det_str} -> {dep_str}"
    
    def __hash__(self):
        return hash((self.determinant, self.dependent))
    
    def __eq__(self, other):
        return (self.determinant == other.determinant and 
                self.dependent == other.dependent)

class InputParser:
    """Handles parsing of functional dependencies and data validation"""
    
    @staticmethod
    def parse_functional_dependencies(fd_text: str) -> List[FunctionDependency]:
        """Parse functional dependencies from text input"""
        fds = []
        lines = [line.strip() for line in fd_text.split('\n') if line.strip()]
        
        for line in lines:
            if '->' not in line:
                continue
            
            try:
                left, right = line.split('->', 1)
                determinant = {attr.strip() for attr in left.split(',')}
                dependent = {attr.strip() for attr in right.split(',')}
                
                # Remove empty strings
                determinant = {attr for attr in determinant if attr}
                dependent = {attr for attr in dependent if attr}
                
                if determinant and dependent:
                    fds.append(FunctionDependency(determinant, dependent))
            except Exception as e:
                st.warning(f"Could not parse FD: {line}")
        
        return fds
    
    @staticmethod
    def validate_data(df: pd.DataFrame, fds: List[FunctionDependency]) -> Tuple[bool, List[str]]:
        """Validate that the data is consistent with functional dependencies"""
        errors = []
        
        # Check if all FD attributes exist in the data
        all_attributes = set(df.columns)
        for fd in fds:
            missing_attrs = (fd.determinant | fd.dependent) - all_attributes
            if missing_attrs:
                errors.append(f"FD '{fd}' references non-existent attributes: {missing_attrs}")
        
        return len(errors) == 0, errors

class FDProcessor:
    """Processes functional dependencies and implements closure algorithms"""
    
    @staticmethod
    def compute_closure(attributes: Set[str], fds: List[FunctionDependency]) -> Set[str]:
        """Compute the closure of a set of attributes under given FDs"""
        closure = set(attributes)
        changed = True
        
        while changed:
            changed = False
            for fd in fds:
                if fd.determinant.issubset(closure) and not fd.dependent.issubset(closure):
                    closure.update(fd.dependent)
                    changed = True
        
        return closure
    
    @staticmethod
    def find_candidate_keys(all_attributes: Set[str], fds: List[FunctionDependency]) -> List[Set[str]]:
        """Find all candidate keys using attribute closure"""
        candidate_keys = []
        
        # Generate all possible attribute combinations, starting from smallest
        for size in range(1, len(all_attributes) + 1):
            for attr_combo in itertools.combinations(all_attributes, size):
                attr_set = set(attr_combo)
                closure = FDProcessor.compute_closure(attr_set, fds)
                
                # If closure covers all attributes, it's a superkey
                if closure == all_attributes:
                    # Check if it's minimal (no proper subset is also a superkey)
                    is_minimal = True
                    for existing_key in candidate_keys:
                        if existing_key.issubset(attr_set):
                            is_minimal = False
                            break
                    
                    if is_minimal:
                        # Remove any existing keys that are supersets of this key
                        candidate_keys = [key for key in candidate_keys if not attr_set.issubset(key)]
                        candidate_keys.append(attr_set)
        
        return candidate_keys
    
    @staticmethod
    def is_prime_attribute(attr: str, candidate_keys: List[Set[str]]) -> bool:
        """Check if an attribute is prime (part of any candidate key)"""
        return any(attr in key for key in candidate_keys)

class NormalFormDetector:
    """Detects the current normal form of a relation"""
    
    @staticmethod
    def is_1nf(df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if relation is in 1NF (atomic values)"""
        for col in df.columns:
            for value in df[col].dropna():
                # Check for common multi-value 
                if isinstance(value, str) and (',' in value or ';' in value or '|' in value):
                    if len(value.split(',')) > 1 or len(value.split(';')) > 1 or len(value.split('|')) > 1:
                        return False, f"Column '{col}' contains multi-valued attributes"
        
        return True, "All attributes are atomic"
    
    @staticmethod
    def is_2nf(df: pd.DataFrame, fds: List[FunctionDependency], 
              candidate_keys: List[Set[str]]) -> Tuple[bool, str]:
        """Check if relation is in 2NF (no partial dependencies)"""
        if not candidate_keys:
            return True, "No candidate keys found"
        
        all_attributes = set(df.columns)
        
        for fd in fds:
            # Check for partial dependencies on candidate keys
            for key in candidate_keys:
                if (fd.determinant.issubset(key) and 
                    fd.determinant != key and
                    len(fd.determinant) < len(key)):
                    
                    # Check if dependent attributes are non-prime
                    non_prime_dependents = [attr for attr in fd.dependent 
                                          if not FDProcessor.is_prime_attribute(attr, candidate_keys)]
                    
                    if non_prime_dependents:
                        return False, f"Partial dependency found: {fd} (non-prime attributes depend on part of candidate key)"
        
        return True, "No partial dependencies found"
    
    @staticmethod
    def is_3nf(df: pd.DataFrame, fds: List[FunctionDependency], 
              candidate_keys: List[Set[str]]) -> Tuple[bool, str]:
        """Check if relation is in 3NF (no transitive dependencies)"""
        all_attributes = set(df.columns)
        
        for fd in fds:
            # Skip trivial FDs and FDs where determinant is a superkey
            if fd.dependent.issubset(fd.determinant):
                continue
            
            is_superkey = any(fd.determinant >= key for key in candidate_keys)
            if is_superkey:
                continue
            
            # Check if all dependent attributes are prime
            all_prime = all(FDProcessor.is_prime_attribute(attr, candidate_keys) 
                           for attr in fd.dependent)
            
            if not all_prime:
                return False, f"Transitive dependency found: {fd} (determinant is not a superkey and dependent has non-prime attributes)"
        
        return True, "No transitive dependencies found"
    
    @staticmethod
    def is_bcnf(df: pd.DataFrame, fds: List[FunctionDependency], 
               candidate_keys: List[Set[str]]) -> Tuple[bool, str]:
        """Check if relation is in BCNF (determinants are superkeys)"""
        for fd in fds:
            # Skip trivial FDs
            if fd.dependent.issubset(fd.determinant):
                continue
            
            # Check if determinant is a superkey
            closure = FDProcessor.compute_closure(fd.determinant, fds)
            is_superkey = closure == set(df.columns)
            
            if not is_superkey:
                return False, f"BCNF violation: {fd} (determinant is not a superkey)"
        
        return True, "All determinants are superkeys"

class DatabaseDecomposer:
    """Handles database decomposition for normalization"""
    
    def __init__(self, df: pd.DataFrame, fds: List[FunctionDependency]):
        self.original_df = df
        self.fds = fds
        self.all_attributes = set(df.columns)
        self.decomposition_steps = []
    
    def normalize_to_1nf(self) -> List[pd.DataFrame]:
        """Normalize to 1NF by handling multi-valued attributes"""
        result_tables = []
        current_df = self.original_df.copy()
        
        for col in current_df.columns:
            multi_value_rows = []
            for idx, value in enumerate(current_df[col]):
                if pd.isna(value):
                    continue
                    
                if isinstance(value, str):
                    # Check for common separators
                    values = None
                    if ',' in value:
                        values = [v.strip() for v in value.split(',')]
                    elif ';' in value:
                        values = [v.strip() for v in value.split(';')]
                    elif '|' in value:
                        values = [v.strip() for v in value.split('|')]
                    
                    if values and len(values) > 1:
                        multi_value_rows.append((idx, values))
            
            if multi_value_rows:
                # Create separate table for multi-valued attribute
                new_rows = []
                for idx, values in multi_value_rows:
                    row_data = current_df.iloc[idx].to_dict()
                    for value in values:
                        new_row = row_data.copy()
                        new_row[col] = value
                        new_rows.append(new_row)
                
                # Remove original multi-value rows and add decomposed rows
                current_df = current_df.drop([idx for idx, _ in multi_value_rows])
                new_df = pd.DataFrame(new_rows)
                current_df = pd.concat([current_df, new_df], ignore_index=True)
                
                self.decomposition_steps.append({
                    'step': '1NF',
                    'reason': f'Decomposed multi-valued attribute: {col}',
                    'table': current_df.copy()
                })
        
        return [current_df]
    
    def normalize_to_2nf(self, tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Normalize to 2NF by removing partial dependencies"""
        result_tables = []
        
        for table in tables:
            candidate_keys = FDProcessor.find_candidate_keys(set(table.columns), self.fds)
            
            if not candidate_keys:
                result_tables.append(table)
                continue
            
            current_tables = [table]
            
            # Find partial dependencies
            for fd in self.fds:
                table_attrs = set(table.columns)
                if not (fd.determinant.issubset(table_attrs) and fd.dependent.issubset(table_attrs)):
                    continue
                
                for key in candidate_keys:
                    if (fd.determinant.issubset(key) and 
                        fd.determinant != key and
                        len(fd.determinant) < len(key)):
                        
                        # Check if dependent attributes are non-prime
                        non_prime_deps = [attr for attr in fd.dependent 
                                        if not FDProcessor.is_prime_attribute(attr, candidate_keys)]
                        
                        if non_prime_deps:
                            # Decompose: create new table with determinant + dependents
                            new_table_attrs = fd.determinant | fd.dependent
                            new_table = table[list(new_table_attrs)].drop_duplicates()
                            
                            # Keep original table without the dependent attributes
                            remaining_attrs = table_attrs - fd.dependent
                            modified_table = table[list(remaining_attrs)].drop_duplicates()
                            
                            current_tables = [modified_table, new_table]
                            
                            self.decomposition_steps.append({
                                'step': '2NF',
                                'reason': f'Removed partial dependency: {fd}',
                                'tables': [t.copy() for t in current_tables]
                            })
                            break
            
            result_tables.extend(current_tables)
        
        return result_tables
    
    def normalize_to_3nf(self, tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Normalize to 3NF by removing transitive dependencies"""
        result_tables = []
        
        for table in tables:
            candidate_keys = FDProcessor.find_candidate_keys(set(table.columns), self.fds)
            current_tables = [table]
            
            # Find transitive dependencies
            for fd in self.fds:
                table_attrs = set(table.columns)
                if not (fd.determinant.issubset(table_attrs) and fd.dependent.issubset(table_attrs)):
                    continue
                
                # Skip if determinant is a superkey
                is_superkey = any(fd.determinant >= key for key in candidate_keys)
                if is_superkey:
                    continue
                
                # Check if dependent attributes are non-prime
                non_prime_deps = [attr for attr in fd.dependent 
                                if not FDProcessor.is_prime_attribute(attr, candidate_keys)]
                
                if non_prime_deps:
                    # Decompose: create new table with determinant + dependents
                    new_table_attrs = fd.determinant | fd.dependent
                    new_table = table[list(new_table_attrs)].drop_duplicates()
                    
                    # Keep original table without the dependent attributes
                    remaining_attrs = table_attrs - fd.dependent
                    modified_table = table[list(remaining_attrs)].drop_duplicates()
                    
                    current_tables = [modified_table, new_table]
                    
                    self.decomposition_steps.append({
                        'step': '3NF',
                        'reason': f'Removed transitive dependency: {fd}',
                        'tables': [t.copy() for t in current_tables]
                    })
                    break
            
            result_tables.extend(current_tables)
        
        return result_tables
    
    def normalize_to_bcnf(self, tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Normalize to BCNF by removing all BCNF violations using an iterative worklist algorithm"""
        worklist = tables.copy()  # Start with the initial tables to check
        final_tables = []         # Store tables that are confirmed to be in BCNF

        while worklist:
            table = worklist.pop(0)  # Get the next table to process
            table_attrs = set(table.columns)
            violation_found = False

            for fd in self.fds:
                # 1. Check if the FD is applicable to the current table
                if not (fd.determinant.issubset(table_attrs) and fd.dependent.issubset(table_attrs)):
                    continue
                
                # 2. Skip trivial dependencies
                if fd.dependent.issubset(fd.determinant):
                    continue
                
                # 3. Check for BCNF violation: 
                closure = FDProcessor.compute_closure(fd.determinant, self.fds)
                is_superkey = closure.issuperset(table_attrs)
                
                if not is_superkey:

                    table1_attrs = fd.determinant.union(fd.dependent)
                    table1 = table[list(table1_attrs)].drop_duplicates().reset_index(drop=True)
                    
                    # Table R2 = attributes of R - Y + X
                    table2_attrs = (table_attrs - fd.dependent).union(fd.determinant)
                    table2 = table[list(table2_attrs)].drop_duplicates().reset_index(drop=True)
                    
                    # Add the two new tables 
                    worklist.append(table1)
                    worklist.append(table2)
                    
                    # Log the step and mark that a violation was handled
                    self.decomposition_steps.append({
                        'step': 'BCNF',
                        'reason': f'Removed BCNF violation: {fd}',
                        'tables': [t.copy() for t in [table1, table2]]
                    })
                    
                    violation_found = True
                    break 
            
            if not violation_found:
                # If we checked all FDs and found no violations, the table is in BCNF.
                final_tables.append(table)
                
        return final_tables

class ERDiagramVisualizer:
    """Creates ER diagrams for database schemas"""
    
    @staticmethod
    def create_er_diagram(tables: List[pd.DataFrame], table_names: List[str] = None) -> graphviz.Digraph:
        """Create an ER diagram for the given tables"""
        dot = graphviz.Digraph(comment='ER Diagram')
        dot.attr(rankdir='TB', size='12,8')
        dot.attr('node', shape='record', style='filled', fillcolor='lightblue')
        
        if table_names is None:
            table_names = [f'Table_{i+1}' for i in range(len(tables))]
        
        # Add tables as nodes
        for i, (table, name) in enumerate(zip(tables, table_names)):
            # Create table representation
            columns = '|'.join([f'{col}' for col in table.columns])
            label = f'{name}|{columns}'
            dot.node(f'table_{i}', label)
        
        return dot
    
    @staticmethod
    def create_normalization_diagram(original_table: pd.DataFrame, 
                                   decomposed_tables: List[pd.DataFrame],
                                   step_name: str) -> graphviz.Digraph:
        """Create a diagram showing normalization step"""
        dot = graphviz.Digraph(comment=f'{step_name} Normalization')
        dot.attr(rankdir='LR', size='14,10')
        
        # Original table
        orig_cols = '|'.join(original_table.columns)
        dot.node('original', f'Original Table|{orig_cols}', 
                shape='record', style='filled', fillcolor='lightcoral')
        
        # Decomposed tables
        for i, table in enumerate(decomposed_tables):
            cols = '|'.join(table.columns)
            dot.node(f'decomp_{i}', f'Table {i+1}|{cols}', 
                    shape='record', style='filled', fillcolor='lightgreen')
            dot.edge('original', f'decomp_{i}', label=step_name)
        
        return dot

class SQLExporter:
    """Exports normalized tables as SQL CREATE statements"""
    
    @staticmethod
    def generate_create_statements(tables: List[pd.DataFrame], 
                                 table_names: List[str] = None) -> str:
        """Generate SQL CREATE TABLE statements"""
        if table_names is None:
            table_names = [f'table_{i+1}' for i in range(len(tables))]
        
        sql_statements = []
        
        for table, name in zip(tables, table_names):
            columns = []
            for col in table.columns:
                # Infer data type based on column data
                sample_data = table[col].dropna()
                if len(sample_data) == 0:
                    col_type = 'VARCHAR(255)'
                elif sample_data.dtype in ['int64', 'int32']:
                    col_type = 'INTEGER'
                elif sample_data.dtype in ['float64', 'float32']:
                    col_type = 'DECIMAL(10,2)'
                else:
                    max_length = sample_data.astype(str).str.len().max()
                    col_type = f'VARCHAR({max(50, int(max_length * 1.2))})'
                
                columns.append(f'    {col} {col_type}')
            
            create_stmt = f"CREATE TABLE {name} (\n" + ',\n'.join(columns) + "\n);"
            sql_statements.append(create_stmt)
        
        return '\n\n'.join(sql_statements)
    
    @staticmethod
    def generate_insert_statements(tables: List[pd.DataFrame], 
                                 table_names: List[str] = None) -> str:
        """Generate SQL INSERT statements with sample data"""
        if table_names is None:
            table_names = [f'table_{i+1}' for i in range(len(tables))]
        
        insert_statements = []
        
        for table, name in zip(tables, table_names):
            if len(table) > 0:
                columns = ', '.join(table.columns)
                insert_stmt = f"INSERT INTO {name} ({columns}) VALUES\n"
                
                values_list = []
                for _, row in table.iterrows():
                    values = []
                    for val in row:
                        if pd.isna(val):
                            values.append('NULL')
                        elif isinstance(val, str):
                            # Escape single quotes in strings
                            escaped_val = str(val).replace("'", "''")
                            values.append(f"'{escaped_val}'")
                        else:
                            values.append(str(val))
                    values_list.append(f"    ({', '.join(values)})")
                
                insert_stmt += ',\n'.join(values_list) + ";"
                insert_statements.append(insert_stmt)
        
        return '\n\n'.join(insert_statements)

def main():
    """Main Streamlit application"""
    st.title("🗃️ Automated Database Normalization Tool")
    st.markdown("Upload your data and functional dependencies to automatically normalize your database!")
    
    # Initialize session state
    if 'normalization_complete' not in st.session_state:
        st.session_state.normalization_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    with st.sidebar:
        st.header("📥 Input Data")
        
        # Data input 
        input_method = st.radio("Choose input method:", ["Upload CSV", "Enter Data Manually"])
        
        df = None
        if input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        else:  # Manual entry
            st.subheader("Manual Data Entry")
            num_cols = st.number_input("Number of columns", min_value=2, max_value=10, value=3)
            num_rows = st.number_input("Number of rows", min_value=1, max_value=20, value=5)

            cols = [f"Col_{i+1}" for i in range(num_cols)]
            data = {}
            
            col_names = st.text_input("Column names (comma-separated)", 
                                    value=", ".join(cols)).split(",")
            col_names = [name.strip() for name in col_names[:num_cols]]
            
            for col in col_names:
                data[col] = []
            
            for row in range(num_rows):
                st.write(f"Row {row + 1}:")
                row_data = {}
                for col in col_names:
                    value = st.text_input(f"{col}", key=f"{col}_{row}", value="")
                    row_data[col] = value if value else None
                for col in col_names:
                    data[col].append(row_data[col])
            
            if st.button("Create DataFrame"):
                df = pd.DataFrame(data)
                st.success("DataFrame created!")
        
        # Functional Dependencies input
        st.header("⚡ Functional Dependencies")
        fd_text = st.text_area(
            "Enter functional dependencies (one per line)",
            placeholder="RollNo -> Name\nRollNo, Subject -> Marks\nName -> Address",
            height=100
        )
        
        # Normalization 
        run_normalization = st.button("🚀 Run Normalization", type="primary")
    
    # Main content area
    if df is not None:
        st.header("📊 Input Data Preview")
        st.dataframe(df, use_container_width=True)
        
        if fd_text.strip():
            st.header("📋 Functional Dependencies")
            fds = InputParser.parse_functional_dependencies(fd_text)
            for fd in fds:
                st.write(f"• {fd}")
        
        if run_normalization and fd_text.strip():
            # Parse functional dependencies
            fds = InputParser.parse_functional_dependencies(fd_text)
            
            if not fds:
                st.error("No valid functional dependencies found!")
                return
            
            # Validate data
            is_valid, errors = InputParser.validate_data(df, fds)
            if not is_valid:
                st.error("Validation errors:")
                for error in errors:
                    st.error(f"• {error}")
                return
            
            # Run normalization process
            with st.spinner("Running normalization analysis..."):
                # Detect current normal form
                detector = NormalFormDetector()
                candidate_keys = FDProcessor.find_candidate_keys(set(df.columns), fds)
                
                # Check normal forms
                is_1nf, nf_1_reason = detector.is_1nf(df)
                is_2nf, nf_2_reason = detector.is_2nf(df, fds, candidate_keys) if is_1nf else (False, "Not in 1NF")
                is_3nf, nf_3_reason = detector.is_3nf(df, fds, candidate_keys) if is_2nf else (False, "Not in 2NF")
                is_bcnf, bcnf_reason = detector.is_bcnf(df, fds, candidate_keys) if is_3nf else (False, "Not in 3NF")
                
                # Determine current normal form
                if is_bcnf:
                    current_nf = "BCNF"
                elif is_3nf:
                    current_nf = "3NF"
                elif is_2nf:
                    current_nf = "2NF"
                elif is_1nf:
                    current_nf = "1NF"
                else:
                    current_nf = "Not in 1NF"
                
                # Perform decomposition
                decomposer = DatabaseDecomposer(df, fds)
                
                # Normalize step by step
                tables_1nf = decomposer.normalize_to_1nf()
                tables_2nf = decomposer.normalize_to_2nf(tables_1nf)
                tables_3nf = decomposer.normalize_to_3nf(tables_2nf)
                tables_bcnf = decomposer.normalize_to_bcnf(tables_3nf)
                
                # Store results in session state
                st.session_state.results = {
                    'original_df': df,
                    'fds': fds,
                    'candidate_keys': candidate_keys,
                    'current_nf': current_nf,
                    'nf_reasons': {
                        '1NF': (is_1nf, nf_1_reason),
                        '2NF': (is_2nf, nf_2_reason),
                        '3NF': (is_3nf, nf_3_reason),
                        'BCNF': (is_bcnf, bcnf_reason)
                    },
                    'decomposition_steps': decomposer.decomposition_steps,
                    'final_tables': tables_bcnf,
                    'tables_by_nf': {
                        '1NF': tables_1nf,
                        '2NF': tables_2nf,
                        '3NF': tables_3nf,
                        'BCNF': tables_bcnf
                    }
                }
                st.session_state.normalization_complete = True
    
    # Display results
    if st.session_state.normalization_complete and st.session_state.results:
        results = st.session_state.results
        
        st.header("🎯 Normalization Results")
        
        # Display current normal form
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Normal Form", results['current_nf'])
        with col2:
            st.metric("Candidate Keys Found", len(results['candidate_keys']))
        with col3:
            st.metric("Final Tables", len(results['final_tables']))
        
        # Show candidate keys
        if results['candidate_keys']:
            st.subheader("🔑 Candidate Keys")
            for i, key in enumerate(results['candidate_keys']):
                st.write(f"Key {i+1}: {{{', '.join(sorted(key))}}}")
        
        # Normal form analysis
        st.subheader("📈 Normal Form Analysis")
        
        for nf in ['1NF', '2NF', '3NF', 'BCNF']:
            is_satisfied, reason = results['nf_reasons'][nf]
            status = "✅" if is_satisfied else "❌"
            st.write(f"{status} **{nf}**: {reason}")
        
        # Show decomposition steps
        if results['decomposition_steps']:
            st.subheader("🔄 Decomposition Steps")
            
            for i, step in enumerate(results['decomposition_steps']):
                with st.expander(f"Step {i+1}: {step['step']} - {step['reason']}"):
                    if 'tables' in step:
                        for j, table in enumerate(step['tables']):
                            st.write(f"**Table {j+1}:**")
                            st.dataframe(table, use_container_width=True)
                    elif 'table' in step:
                        st.dataframe(step['table'], use_container_width=True)
        
        # Final normalized tables
        st.subheader("📋 Final Normalized Tables (BCNF)")
        
        for i, table in enumerate(results['final_tables']):
            st.write(f"**Table {i+1}:**")
            st.dataframe(table, use_container_width=True)
        
        # Visualization
        st.subheader("📊 ER Diagram Visualization")
        
        try:
            visualizer = ERDiagramVisualizer()
            table_names = [f"Table_{i+1}" for i in range(len(results['final_tables']))]
            er_diagram = visualizer.create_er_diagram(results['final_tables'], table_names)
            st.graphviz_chart(er_diagram)
        except Exception as e:
            st.error(f"Error generating ER diagram: {str(e)}")
        
        # Export options
        st.subheader("💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # SQL export
            st.write("**SQL Export**")
            sql_exporter = SQLExporter()
            create_statements = sql_exporter.generate_create_statements(results['final_tables'], table_names)
            insert_statements = sql_exporter.generate_insert_statements(results['final_tables'], table_names)
            
            sql_content = create_statements + "\n\n" + insert_statements
            sql_b64 = base64.b64encode(sql_content.encode()).decode()
            
            st.download_button(
                label="Download SQL Script",
                data=sql_content,
                file_name="normalized_tables.sql",
                mime="text/plain"
            )
        
        with col2:
            # CSV export
            st.write("**CSV Export**")
            
            # Create a zip file containing all tables as CSVs
            csv_buffer = io.BytesIO()
            with zipfile.ZipFile(csv_buffer, 'w') as zf:
                for i, table in enumerate(results['final_tables']):
                    csv_data = table.to_csv(index=False)
                    zf.writestr(f"{table_names[i]}.csv", csv_data)
            
            csv_buffer.seek(0)
            
            st.download_button(
                label="Download CSV Files (ZIP)",
                data=csv_buffer,
                file_name="normalized_tables.zip",
                mime="application/zip"
            )
        
        with col3:
            # ER Diagram export
            st.write("**ER Diagram Export**")
            
            try:
                # Create ER diagram for final tables
                er_diagram = visualizer.create_er_diagram(results['final_tables'], table_names)
                
                # Render to PNG format
                er_diagram_png = er_diagram.pipe(format='png')
                
                st.download_button(
                    label="Download ER Diagram (PNG)",
                    data=er_diagram_png,
                    file_name="er_diagram.png",
                    mime="image/png"
                )
                
                # Render to SVG format
                er_diagram_svg = er_diagram.pipe(format='svg')
                
                st.download_button(
                    label="Download ER Diagram (SVG)",
                    data=er_diagram_svg,
                    file_name="er_diagram.svg",
                    mime="image/svg+xml"
                )
            except Exception as e:
                st.error(f"Error generating ER diagram for export: {str(e)}")

if __name__ == "__main__":
    main()