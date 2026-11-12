# HOSxP export joins

These source-system queries support the canonical CSV export. They are reference examples for the hospital exporter, not commands run by `bba`. Apply the approved adult-cohort and access controls before staging data. Raw and encrypted clinical exports stay outside version control. Validate the resulting bundle against the [current ingest schemas](../src/bba/ingest/schemas.py).


These examples document how to pull the supporting HOSxP tables for the
adult inpatient transfusion subpopulation. Replace `DATE '<START_DATE>'`
and `DATE '<END_DATE>'` with the export window, for example the first and
last date of the audit month or year.

Common cohort pattern:

- Anchor on `CUH.BDVST` because it contains blood-bank visits/orders.
- Join `CUH.PT` on `HN` to compute adult age at `BDVSTDATE`.
- Keep inpatient rows only: `AN IS NOT NULL`.
- Exclude cancelled blood-bank visits: `CANCELDATE IS NULL`.
- Join most clinical evidence tables back to the subpopulation by `AN`.

```sql
/* LAB */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN --,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,sp.AN
--    ,e.NAME AS SEX
--    ,FLOOR(MONTHS_BETWEEN(sp.BDVSTDATE,p.BRTHDATE)/12) AS AGE
--    ,p.BRTHDATE
--    ,OCCPTN
    ,l.LVSTDATE
    ,l.LVSTTIME
    ,l.LABGRP
    ,(SELECT NAME FROM CUH.LABGRP b WHERE b.LABGRP = l.LABGRP) AS NAME_LABGRP
    ,l.LABEXM
    ,(SELECT NAME FROM CUH.LABEXM x WHERE x.LABEXM = l.LABEXM) AS NAME_LABEXM
    ,l.RESULT
    ,l.MINNRM
    ,l.MAXNRM
    ,l.NRMUNIT
FROM SUBPOPULATION sp
--    LEFT JOIN CUH.PT p ON p.HN = sp.HN
--    LEFT JOIN CUH.MALE e ON e.MALE = p.MALE
    LEFT JOIN CUH.LVSTEXM l ON l.AN = sp.AN
WHERE l.LVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
    AND l.RESULT IS NOT NULL;
```

```sql
/* Diagnosis */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN ,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT
    sp.HN
    ,sp.AN
    ,sp.BDVSTDATE AS V_DATE
    ,d2.NAME AS NAME_DIAGTYPE
    ,d.ICD10
    ,c.ICD10WHO
    ,c.NAME AS NAME_ICD10
    ,'IN_HOSPITAL' AS DIAG_TYPE
FROM SUBPOPULATION sp
    INNER JOIN CUH.IPTSUMDIAG d ON d.AN = sp.AN
    LEFT JOIN CUH.ICD10 c ON c.ICD10 = d.ICD10
    LEFT JOIN CUH.DIAGTYPE d2 ON d2.DIAGTYPE = d.DIAGTYPE
WHERE sp.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>';
--    AND d.DIAGTYPE = 3
--    AND c.ICD10WHO NOT IN ('O82','O820','O821','O842')
```

```sql
/* MED */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN ,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,sp.AN
    ,p.PRSCDATE
    ,p.PRSCTIME
    ,p2.MEDITEM
    ,m.MEDNAME AS NAME_MEDITEM
    ,r.GENERIC
    ,r.NAME AS NAME_GENERIC
    ,m.STRENGTH
    ,(SELECT NAME FROM CUH.STRENGTHUNIT s WHERE s.STRENGTHUNIT = m.STRENGTHUNIT) AS STRENGTHUNIT
    ,(SELECT NAME FROM CUH.MEDUSETYPE m3 WHERE m3.MEDFORM = m.MEDFORM AND m3.MEDUSETYPE = m.MEDUSETYPE) AS MEDUSETYPE
    ,(SELECT NAME FROM CUH.MEDUSEQTY m2 WHERE m2.MEDUSEQTY = m.MEDUSEQTY) AS MEDUSEQTY
FROM SUBPOPULATION sp
    INNER JOIN CUH.PRSC p ON p.AN = sp.AN
    LEFT JOIN CUH.PRSCDT p2 ON p2.SPHMLCT = p.SPHMLCT AND p2.PRSCNO = p.PRSCNO
    LEFT JOIN CUH.MEDITEMDIS m ON m.MEDITEM = p2.MEDITEM
    LEFT JOIN CUH.MEDITEMGNR g ON g.MEDITEM = m.MEDITEM
    LEFT JOIN CUH.MEDGENERIC r ON r.GENERIC = g.GENERIC
WHERE p.PRSCDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
    AND m.MEDTYPE = 10;
```

```sql
/* Patient Background */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN ,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,sp.AN
    ,d.DLVSTREQNO
    ,d.DLVSTRECDATE
    ,d.DLVSTRECTIME
    ,d.WARD
    ,d.ROOM
    ,d.BEDNO
    ,d.PTNTTYPE
    ,d.GRVDTY
    ,d.PARITY
    ,d.PARITY2
    ,d.PARITY3
    ,d.PARITY4
    ,d.ABRTN
    ,d.ABRTNFLAG
    ,d.CURTTFLAG
    ,d.ANCFLAG
    ,d.ANCPLACE
    ,(SELECT ANCPLACENM FROM CUH.ANCPLACE a WHERE a.ANCPLACE = d.ANCPLACE) AS NAME_ANCPLACE
    ,d.ANCPLACE2
    ,(SELECT ANCPLACENM FROM CUH.ANCPLACE a WHERE a.ANCPLACE = d.ANCPLACE2) AS NAME_ANCPLACE2
    ,d.NOATTEND
    ,d.NOATTEND2
    ,d.LMPDATE
    ,d.EDCDATE
    ,d.GSTTNAGE
    ,d.GSTTNDAY
    ,d.LABFLAG
    ,d.TKDATEBLDG
    ,d.BLDTEST
    ,d.RHTEST
    ,(SELECT NAME FROM CUH.RHTEST r WHERE r.RHTEST = d.RHTEST) AS NAME_RHTEST
    ,d.TKDATEHBSAG
    ,d.HBSAGTEST
--    ,(SELECT NAME FROM CUH.HBSAGTEST h WHERE h.HBSAGTEST = d.HBSAGTEST) AS NAME_HBSAGTEST
    ,d.TKDATEVDRL
    ,d.VDRLTEST
--    ,(SELECT NAME FROM CUH.VDRLTEST v WHERE v.VDRLTEST = d.VDRLTEST) AS NAME_VDRLTEST
    ,d.TKDATETPHA
    ,d.TPHATEST
    ,(SELECT NAME FROM CUH.TPHATEST t WHERE t.TPHATEST = d.TPHATEST) AS NAME_TPHATEST
    ,d.HIVTEST
    ,(SELECT NAME FROM CUH.HIVTEST h2 WHERE h2.HIVTEST = d.HIVTEST) AS NAME_HIVTEST
    ,d.TKDATEHCT
    ,d.HCTTEST
    ,d.TKDATECMIA
    ,d.CMIATEST
    ,(SELECT NAME FROM CUH.CMIATEST c WHERE c.CMIATEST = d.CMIATEST) AS NAME_CMIATEST
    ,d.LEADSIGN
--    ,(SELECT NAME FROM CUH.LEADSIGN l WHERE l.LEADSIGN = d.LEADSIGN) AS NAME_LEADSIGN
    ,d.LEADSIGNOTH
    ,d.HEADENGGMNT
    ,d.HEARTSOUND
    ,d.SOUNDDATE
    ,d.SOUNDTIME
    ,d.RECSTATUS
    ,d.KEYBACK
    ,d.TKDATEHIV
    ,d.FHRAMT
    ,d.DLVSTPLACE
    ,d.CMIATESTNEW
    ,d2.INDCTION
    ,d2.INDCTIONOTH
    ,d2.EPISTM
    ,d2.MTHDRP
    ,d2.PLCNTDL
    ,d3.ONSETDATE
    ,d3.ONSETTIME
    ,d3.FDLTTNDATE
    ,d3.FDLTTNTIME
    ,d3.DBRTHDATE
    ,d3.DBRTHTIME
    ,d3.PLCTDATE
    ,d3.PLCTTIME
    ,d3.RPTRTYPE
    ,d3.RPTRDATE
    ,d3.RPTRTIME
FROM SUBPOPULATION sp
    LEFT JOIN CUH.DLVSTEXT d ON d.AN = sp.AN
    LEFT JOIN CUH.DLVSTDESC d2 ON d2.DLVSTREQNO = d.DLVSTREQNO
    LEFT JOIN CUH.DLVST d3 ON d3.DLVSTREQNO = d.DLVSTREQNO
WHERE d.DLVSTRECDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>';
```

```sql
/* IPDNRFOCUSDT */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN ,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,sp.AN
    ,i.ITEMNO
    ,i.ITEMNO2
    ,i.FOCUS
    ,i.ACTION
    ,i.RESPONSE
    ,i.PROGRESSDATE
    ,i.PROGRESSTIME
FROM SUBPOPULATION sp
    INNER JOIN CUH.IPDNRFOCUSDT i ON i.AN = sp.AN
WHERE i.CANCELDATE IS NULL;
```

```sql
/* IPDADMPROGRESS */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN --,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,i.*
FROM SUBPOPULATION sp
    INNER JOIN CUH.IPDADMPROGRESS i ON i.AN = sp.AN
WHERE i.CANCELDATE IS NULL;
```

```sql
/* INCPT */
SELECT
  "KCMH_HIS"."INCPT"."hn" AS "hn",
  "KCMH_HIS"."INCPT"."incdate" AS "incdate",
  "KCMH_HIS"."INCPT"."inctime" AS "inctime",
  "KCMH_HIS"."INCPT"."ordercode" AS "ordercode",
  "KCMH_HIS"."INCPT"."income" AS "income",
  "KCMH_HIS"."INCPT"."an" AS "an",
  "KCMH_HIS"."INCPT"."canceldate" AS "canceldate",
  "KCMH_HIS"."INCPT"."incgrp" AS "incgrp",
  "Incgrp"."name" AS "Incgrp__name",
  "Incgrp"."incgrp" AS "Incgrp__incgrp",
  "Oprtact - Income"."oprtact" AS "Oprtact - Income__oprtact",
  "Oprtact - Income"."name_en" AS "Oprtact - Income__name_en",
  "Oprtact - Income"."name" AS "Oprtact - Income__name",
  "Oprtact - Income"."income" AS "Oprtact - Income__income",
  "Oprtact - Income"."maxamt" AS "Oprtact - Income__maxamt",
  "Oprtact - Income"."dftamt" AS "Oprtact - Income__dftamt",
  "Oprtact - Income"."rfndamt" AS "Oprtact - Income__rfndamt",
  "Oprtact - Income"."minamt" AS "Oprtact - Income__minamt",
  "Oprtact - Income"."spclty" AS "Oprtact - Income__spclty",
  "Oprtact - Income"."otamt" AS "Oprtact - Income__otamt",
  "Oprtact - Income"."otrfndamt" AS "Oprtact - Income__otrfndamt",
  "Oprtact - Income"."icd9cm" AS "Oprtact - Income__icd9cm",
  "Oprtact - Income"."icd10tm" AS "Oprtact - Income__icd10tm",
  "Oprtact - Income"."oprttype" AS "Oprtact - Income__oprttype",
  "Oprtact - Income"."cost" AS "Oprtact - Income__cost",
  "Oprtact - Income"."nhso" AS "Oprtact - Income__nhso",
  "Oprtact - Income"."drgmx" AS "Oprtact - Income__drgmx",
  "Oprtact - Income"."csmbs" AS "Oprtact - Income__csmbs",
  "Oprtact - Income"."sss" AS "Oprtact - Income__sss",
  "Oprtact - Income"."var" AS "Oprtact - Income__var",
  "Oprtact - Income"."minvalue" AS "Oprtact - Income__minvalue",
  "Oprtact - Income"."maxvalue" AS "Oprtact - Income__maxvalue",
  "Oprtact - Income"."dscper" AS "Oprtact - Income__dscper",
  "Oprtact - Income"."dscamt" AS "Oprtact - Income__dscamt",
  "Oprtact - Income"."prfact" AS "Oprtact - Income__prfact",
  "Oprtact - Income"."sumprfamt" AS "Oprtact - Income__sumprfamt",
  "Oprtact - Income"."oprtgrp" AS "Oprtact - Income__oprtgrp",
  "Oprtact - Income"."icdnum" AS "Oprtact - Income__icdnum",
  "Oprtact - Income"."icdtype" AS "Oprtact - Income__icdtype",
  "Oprtact - Income"."highcost" AS "Oprtact - Income__highcost",
  "Oprtact - Income"."major" AS "Oprtact - Income__major",
  "Oprtact - Income"."dz8" AS "Oprtact - Income__dz8",
  "Oprtact - Income"."vac" AS "Oprtact - Income__vac",
  "Oprtact - Income"."vcctype" AS "Oprtact - Income__vcctype",
  "Oprtact - Income"."timemake" AS "Oprtact - Income__timemake",
  "Oprtact - Income"."validcode" AS "Oprtact - Income__validcode",
  "Oprtact - Income"."canceldate" AS "Oprtact - Income__canceldate",
  "Oprtact - Income"."datasource" AS "Oprtact - Income__datasource",
  "Oprtact - Income"."firststf" AS "Oprtact - Income__firststf",
  "Oprtact - Income"."firstdate" AS "Oprtact - Income__firstdate",
  "Oprtact - Income"."laststf" AS "Oprtact - Income__laststf",
  "Oprtact - Income"."lastdate" AS "Oprtact - Income__lastdate",
  "Oprtact - Income"."incomeold" AS "Oprtact - Income__incomeold",
  "Oprtact - Income"."icd9cmadd1" AS "Oprtact - Income__icd9cmadd1",
  "Oprtact - Income"."icd9cmadd2" AS "Oprtact - Income__icd9cmadd2",
  "Oprtact - Income"."oprtactauto" AS "Oprtact - Income__oprtactauto",
  "Oprtact - Income"."perday" AS "Oprtact - Income__perday",
  "Oprtact - Income"."unit" AS "Oprtact - Income__unit",
  "Oprtact - Income"."note" AS "Oprtact - Income__note",
  "Oprtact - Income"."cancelstf" AS "Oprtact - Income__cancelstf",
  "Oprtact - Income"."cancelnote" AS "Oprtact - Income__cancelnote",
  "Oprtact - Income"."oprtactgrp" AS "Oprtact - Income__oprtactgrp",
  "Oprtact - Income"."thtype" AS "Oprtact - Income__thtype",
  "Oprtact - Income"."addper" AS "Oprtact - Income__addper",
  "Oprtact - Income"."remflag" AS "Oprtact - Income__remflag",
  "Oprtact - Income"."nodefchk" AS "Oprtact - Income__nodefchk"
FROM
  "KCMH_HIS"."INCPT"

LEFT JOIN "KCMH_HIS"."INCGRP" AS "Incgrp" ON "KCMH_HIS"."INCPT"."incgrp" = "Incgrp"."incgrp"
  LEFT JOIN "cleaned_ddc_internal"."OPRTACT" AS "Oprtact - Income" ON "KCMH_HIS"."INCPT"."income" = "Oprtact - Income"."oprtact"
WHERE
  "KCMH_HIS"."INCPT"."incgrp" BETWEEN 110
   AND 111
  AND (
    "KCMH_HIS"."INCPT"."incdate" >= timestamp '2025-01-01 00:00:00.000'
  )
  AND (
    "KCMH_HIS"."INCPT"."incdate" < timestamp '2026-01-01 00:00:00.000'
  )
  AND ("KCMH_HIS"."INCPT"."an" IS NOT NULL)
  AND (
    ("KCMH_HIS"."INCPT"."an" <> '')

    OR ("KCMH_HIS"."INCPT"."an" IS NULL)
  )
```
