"""
This module facilitates the creation of the ontologies used for annotating tables.
Ontologies currently in scope: Schema.org and DBpedia.
Each ontology is preprocessed to meet a shared format, and stored in a DataFrame as well as dictionary.
"""

import datetime
import json
import pickle
import requests
import typing

import numpy as np
import pandas as pd

from SPARQLWrapper import SPARQLWrapper, JSON


def build_schema_ontology(ontology_dir: str):
    properties = (
        json.loads(
            requests
            .get("https://schema.org/version/latest/schemaorg-current-https.jsonld")
            .content
        )
    )
    
    df_dict = {}
    i = 0
    for prop_dict in properties["@graph"]:
        _type = prop_dict["@type"]
        _comment = prop_dict["rdfs:comment"]
        _id = prop_dict["@id"]
        _cleaned_label = prop_dict["rdfs:label"]
        if isinstance(_cleaned_label, typing.Dict):
            _cleaned_label = prop_dict["rdfs:label"]["@value"]
        _superclasses_list = np.nan
        _domains_list = np.nan
        _ranges_list = np.nan
        _superproperties_list = np.nan
        if "rdfs:subClassOf" in prop_dict:
            _superclasses_list = _create_item_list(prop_dict, "rdfs:subClassOf")
        if "schema:domainIncludes" in prop_dict:
            _domains_list = _create_item_list(prop_dict, "schema:domainIncludes")
        if "schema:rangeIncludes" in prop_dict:
            _ranges_list = _create_item_list(prop_dict, "schema:rangeIncludes")
        if "rdfs:subPropertyOf" in prop_dict:
            _superproperties_list = _create_item_list(prop_dict, "rdfs:subPropertyOf")

        df_dict.update(
            {
                i:
                (_id, _cleaned_label, _comment, _type, _domains_list, _ranges_list, _superclasses_list, _superproperties_list)
            }
        )
        i+=1

    properties_df = pd.DataFrame.from_dict(
        df_dict,
        orient="index",
        columns=["id", "cleaned_label", "description", "type", "domain", "range", "superclasses", "superproperties"]
    )
    
    prop_processed_df = (
        properties_df
        .reset_index(drop=True)
        .pipe((_split_string_on_case, "df"), col="cleaned_label")
        .sort_values(by="cleaned_label")
        .drop_duplicates(subset=["cleaned_label"], keep="first")
        .replace(
            {"nan": None, np.nan: None}
        )
    )
    # One change is made to this ontology: 
    # the cleaned label of identifier is taken as "id" instead of "identfier"
    # "id" is one of the most occurring column names and it is well known to represent an "identifier".
    prop_processed_df.loc[
        prop_processed_df["id"] == "schema:identifier", "cleaned_label"
    ] = "id"
    
    prop_processed_dict = (
        prop_processed_df
        .set_index("cleaned_label", drop=False)
        .to_dict(orient="index")
    )
    
    _write_ontology(prop_processed_df, prop_processed_dict, "schema", ontology_dir)
    
    return prop_processed_df, prop_processed_dict


def build_dbpedia_ontology(ontology_dir: str):
    
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    offset = 0
    properties = pd.DataFrame()
    df = pd.DataFrame({"a": ["b"]})
    while not df.empty:
        query = (
            f"""
            select distinct ?property ?range ?domain ?description ?superproperty where {{
                ?property a rdf:Property .
                OPTIONAL{{?property rdfs:domain ?domain .}}
                OPTIONAL{{?property rdfs:comment ?description . FILTER (langMatches(lang(?description),"en"))}}
                OPTIONAL{{?property rdfs:range ?range .}}
                OPTIONAL{{?property rdfs:subPropertyOf ?superproperty .}}
            }}
            ORDER BY ?property
            OFFSET {offset}
            """
        )
        # DBpedia limits results to 10000 rows per query
        offset += 10000
        sparql.setQuery(query)
        df = pd.json_normalize(
            sparql.query().convert()
            ['results']['bindings']
        )
        properties = properties.append(
            df[df.columns[df.columns.str.endswith(".value")]]
        )
        
    prop_processed_df = (
        properties
        [properties["property.value"].str.startswith("http://dbpedia.org/ontology")]
        .reset_index(drop=True)
        .pipe((_add_cleaned_label, "df"), col="property.value")
        .pipe((_split_string_on_char, "df"), col="cleaned_label")
        .pipe((_split_string_on_case, "df"), col="cleaned_label")
        .replace(
            {np.nan: None, "nan": None}
        )
        .rename(
            {
                "property.value": "id",
                "domain.value": "domain",
                "description.value": "description",
                "range.value": "range",
                "superproperty.value": "superproperty"
            },
            axis=1
        )
        # some labels have multiple domains, this aggregation groups them into a list
        .pivot_table(index="cleaned_label", aggfunc=list).applymap(
            lambda x: x[0] if len(set(x))==1 else x
        )
        .reset_index()
    )
    # Add list to ensure consistency with schema.org formatting.
    prop_processed_df[["superproperty", "domain", "range"]] = prop_processed_df[["superproperty", "domain", "range"]].applymap(
        lambda x: [x] if isinstance(x, str) else x
    )
    
    prop_processed_dict = (
        prop_processed_df
        .set_index("cleaned_label", drop=False)
        .to_dict(orient="index")
    )
    
    _write_ontology(prop_processed_df, prop_processed_dict, "dbpedia", ontology_dir)
    
    return prop_processed_df, prop_processed_dict


def _split_string_on_char(df, col):
    df[col] = df[col].str.split(r"[/#:]").str[-1]

    return df


def _split_string_on_case(df, col):
    split_col = df[col].astype(str).str.findall("[0-9,a-z,.,\"#!$%\^&\*;:{}=\-_`~()\n\t\d]+|[A-Z](?:[A-Z]*(?![a-z])|[a-z]*)").explode()
    df[col] = split_col.astype(str).str.lower().groupby(level=0).agg(' '.join)

    return df


def _add_cleaned_label(df, col):
    return df.assign(cleaned_label=df[col])


def _write_ontology(ontology_df: pd.DataFrame, ontology_dict: typing.Dict, ontology_name: str, ontology_dir: str):
    date = datetime.date.today().strftime("%Y%m%d")
    filepath = f"{ontology_dir}{ontology_name}_{date}"
    
    ontology_df.to_pickle(f"{filepath}.pkl")
    
    with open(f"{filepath}.pickle", "wb+") as f:
        pickle.dump(ontology_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_ontology(ontology_name: str, ontology_dir: str, date: str = None):
    
    if not isinstance(date, str):
        date = datetime.date.today().strftime("%Y%m%d")
    filepath = f"{ontology_dir}{ontology_name}_{date}"
    
    ontology_df = pd.read_pickle(f"{filepath}.pkl")
    
    with open(f"{filepath}.pickle", "rb") as f:
        ontology_dict = pickle.load(f)

    return ontology_df, ontology_dict


def _create_item_list(prop_dict, item_type):
    prop_items = prop_dict[item_type]
    
    if not isinstance(prop_items, list):
        prop_items = [prop_items]
        
    return [prop_item["@id"] for prop_item in prop_items]
