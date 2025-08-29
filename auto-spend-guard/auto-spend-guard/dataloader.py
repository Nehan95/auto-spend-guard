import pandas as pd
import json
import os
from typing import Dict, Any
from pathlib import Path

class DataLoader:
    """Loads data files from the docs folder into pandas DataFrames"""
    
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.dataframes = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data files from the docs folder"""
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Docs path {self.docs_path} does not exist")
            
        # Load CSV files first
        for csv_file in self.docs_path.glob("*.csv"):
            df_name = csv_file.stem
            self.dataframes[df_name] = pd.read_csv(csv_file)
            print(f"Loaded {df_name}: {csv_file.name} with shape {self.dataframes[df_name].shape}")
            
        # Load JSON files with unique names
        for json_file in self.docs_path.glob("*.json"):
            df_name = json_file.stem
            # If CSV with same name exists, append suffix to JSON
            if df_name in self.dataframes:
                df_name = f"{df_name}-json"
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Convert JSON to DataFrame - handle different JSON structures
            if isinstance(data, list):
                self.dataframes[df_name] = pd.DataFrame(data)
            elif isinstance(data, dict):
                # If it's a nested structure, try to normalize it
                if 'vendors' in data and isinstance(data['vendors'], list):
                    # Special handling for vendor data structure
                    vendors_df = pd.DataFrame(data['vendors'])
                    # Flatten nested objects for better analysis
                    flattened_vendors = []
                    for vendor in data['vendors']:
                        flat_vendor = {
                            'vendor_id': vendor.get('vendor_id'),
                            'vendor_name': vendor.get('vendor_name'),
                            'vendor_type': vendor.get('vendor_type'),
                            'annual_budget_approved': vendor.get('annual_budget_approved'),
                            'current_annual_spend': vendor.get('current_annual_spend'),
                            'budget_variance': vendor.get('budget_variance'),
                            'usage_purpose': vendor.get('usage_purpose'),
                            'business_justification': vendor.get('business_justification'),
                            'owner_name': vendor.get('owner', {}).get('name'),
                            'owner_title': vendor.get('owner', {}).get('title'),
                            'owner_department': vendor.get('owner', {}).get('department'),
                            'owner_email': vendor.get('owner', {}).get('email'),
                            'approver_name': vendor.get('approver', {}).get('name'),
                            'approver_title': vendor.get('approver', {}).get('title'),
                            'status': vendor.get('status'),
                            'risk_level': vendor.get('risk_level'),
                            'contract_end_date': vendor.get('contract_details', {}).get('end_date'),
                            'renewal_date': vendor.get('contract_details', {}).get('renewal_date')
                        }
                        flattened_vendors.append(flat_vendor)
                    self.dataframes[df_name] = pd.DataFrame(flattened_vendors)
                else:
                    # For other JSON structures, use json_normalize
                    self.dataframes[df_name] = pd.json_normalize(data)
            
            print(f"Loaded {df_name}: {json_file.name} with shape {self.dataframes[df_name].shape}")
            
        return self.dataframes
    
    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Get a specific DataFrame by name"""
        if name not in self.dataframes:
            raise KeyError(f"DataFrame '{name}' not found. Available: {list(self.dataframes.keys())}")
        return self.dataframes[name]
    
    def get_available_datasets(self) -> list:
        """Get list of available dataset names"""
        return list(self.dataframes.keys())
    
    def get_dataframe_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific DataFrame"""
        df = self.get_dataframe(name)
        return {
            "name": name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "sample_data": df.head(3).to_dict('records')
        }
    
    def search_data(self, query: str, dataset_name: str = None) -> Dict[str, Any]:
        """Search for data across datasets"""
        results = {}
        
        if dataset_name:
            datasets = {dataset_name: self.get_dataframe(dataset_name)}
        else:
            datasets = self.dataframes
            
        for name, df in datasets.items():
            # Simple text search across all columns
            mask = df.astype(str).apply(lambda x: x.str.contains(query, case=False, na=False)).any(axis=1)
            if mask.any():
                results[name] = {
                    "matches": df[mask].to_dict('records'),
                    "count": mask.sum()
                }
                
        return results
