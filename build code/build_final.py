# Basic packages
import numpy as np
import pandas as pd
import gzip, gc, re

# Pandas display options
pd.options.display.max_columns = 200
pd.options.display.max_rows = 1000
pd.set_option('max_info_columns', 200)
pd.set_option('max_colwidth',1000)
pd.set_option('display.width',None)

# Numpy display options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

# Paths
home = "/accounts/projects/crwalters/cncrime"
data_path = "/accounts/projects/crwalters/cncrime/data/NCERDC"
firstyr = 2006
lastyr = 2013

### 1) Build CMB files directly
if 0:
    print("\n\nWorking on CMB files")
    finals = []
    for year in range(2006,2014):
        print("Loading course membership for {}".format(year))
        with gzip.open(data_path + "/Student/Course Membership/crs_memb_pub{}.dta.gz".format(year,str(year)[-2:]), 'rb') as inF:
            cmb = pd.read_stata(inF)

        # Clean coursetitle to make more uniform over years
        cmb['coursetitle'] = cmb.coursetitle.str.lower()
        cmb['coursetitle'] = cmb.coursetitle.str.strip()
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'(ma)$','math')
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'(eng)$','english')
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'( adv )',' advanced ')
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'(self contained)','self-contained')
        cmb['coursetitle'] = cmb.coursetitle.str.replace(r'([\s]+)',' ')

        # Set of courses taken...
            # these won't be year specific, so need to recode later
        # cmb['unique_courses'] = cmb.groupby(['lea','schlcode','mastid']).coursetitle.apply(lambda x: str(x.unique()))
        # cmb['unique_courses'] = cmb.unique_courses.astype(str)

        # Subset to the state courses we want
        cmb['coursecode'] = pd.to_numeric(cmb.statecourse.str.replace('[^0-9]','').apply(lambda x: x[:4]))
        cmb['localccode'] = cmb.localcourse.str.extract('^([0-9]{4,5})')
        courses = [0,1010,1021,1022,1023,1024,2001,2003,2021,2022,2023,2024,2030,2025]
        cmb = cmb.loc[cmb.coursecode.isin(courses)]

        # Subset to the grades we want
        cmb = cmb.loc[pd.to_numeric(cmb.grade, errors='coerce').between(3,12)]

        # Drop unnecessary columns
        for col in ['numstudents','nstudents','reporting_year','collection_code'] + ['localcourse','statecourse','course_cycle','meetingcode']:
            try:
                cmb.drop(col, axis=1, inplace=True)
            except:
                pass

        # Add to aggregator
        cmb['year'] = year
        cmb = cmb.drop_duplicates()
        finals += [cmb.copy(),]
        del cmb
        gc.collect()

    # Put together CMB build
    data = pd.concat(finals, sort=False, ignore_index=True)
    data = data.loc[data.mastid.notnull()]
    data['grade'] = pd.to_numeric(data.grade)

    # Fix coding of ethnicity
    data['ethnic'] = data.ethnic.replace({1:"I",2:"A",3:"H",4:"B",5:"W",6:"M"})
    data['ethnic'] = data.ethnic.replace({'1':"I",'2':"A",'3':"H",'4':"B",'5':"W",'6':"M"})

    # # Reformat unique courses
    # data['unique_courses'] = data.unique_courses.str.replace("' '","','")
    # data['unique_courses'] = data.unique_courses.str.replace("'\n '","','")
    # data['unique_courses'] = data.unique_courses.apply(lambda x: sorted(eval(x))).astype(str)
    # data['unique_courses_id'] = data.groupby(['lea','schlcode','year','unique_courses']).grouper.group_info[0]

    # Subset data
    data = data[['mastid','year','coursetitle','lea','schlcode','teachid','grade','coursecode','localccode',
                    'sex','ethnic','birthdt','semester','section']] # 'unique_courses','unique_courses_id'
    data['classroom_id'] = data.groupby(['lea','schlcode','year','grade','coursetitle','semester','section']).grouper.group_info[0]
    data['school_course_grade_year_fe'] = data.groupby(['lea','schlcode','year','grade','coursetitle']).grouper.group_info[0]

    # Save full dataset
    data.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/cmb.pkl.gz", compression='gzip')


### 1b) Add p values
if 0:
    from scipy.stats import chi2_contingency
    data = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/cmb.pkl.gz", compression='gzip')

    # Add pvalues for grades 4-8 and 2007 on
    def chi2(table):
        for col in table.columns:
            if table[col].sum() == 0:
                table = table.drop(col, axis=1)
        table = table.values
        r, p, dof, e = chi2_contingency(table)
        return p

    def classroom_chi2(c):
        current = data.loc[data.school_course_grade_year_fe == c]
        lag = data.loc[(data.year == current.year.max() - 1) &
                            data.mastid.isin(current.mastid), ['mastid','classroom_id']].rename(
                    columns={'classroom_id':'lag_classrom_id'})
        current = current[['mastid','classroom_id']].merge(lag, how='inner', on='mastid')
        table = pd.crosstab(current.classroom_id, current.lag_classrom_id)
        try:
            p = chi2(table)
        except:
            p = None
        print(c,table,p,"\n", flush=True)
        return c, p

    gc.collect()
    import multiprocessing
    if __name__ == "__main__":
        p = multiprocessing.Pool(8)
        ans = p.map(classroom_chi2, data.loc[(data.year >= 2007) & data.grade.between(4,12)].school_course_grade_year_fe.unique())

    pvals = pd.DataFrame(ans, columns=['school_course_grade_year_fe','pval'])
    data = data.merge(pvals, how='left', on=['school_course_grade_year_fe'])
    data.loc[data.pval.notnull()].groupby(['lea','schlcode','year','grade','coursetitle']).pval.first().reset_index().to_pickle(
        "/accounts/projects/crwalters/cncrime/users/ekrose/data/pvals.pkl.gz", compression='gzip')

### 3) Build EOG data
if 0:
    print("\n\nAdding EOG files")
    eogs = []
    for year in range(1996,2014):
        for grade in range(3,9):
            if ((year == 1996) & (grade >= 5)):
                continue
            print("Loading EOG for {} grade {}".format(year,grade))
            with gzip.open(data_path + "/Student/End of Grade/{}/eog{}pub{}.dta.gz".format(year,grade,str(year)[-2:]), 'rb') as inF:
                eog = pd.read_stata(inF)

            # Subset to non duplicated key fars
            tmp = eog[['mastid','lea','schlcode','sex','ethnic','bdate']].drop_duplicates()
            tmp['year'] = year
            tmp['grade'] = grade

            # Find best math
            if 'administ' not in eog.columns:
                eog['administ'] = "0"
            tokeep = ['mastid','mathscal','tchmtjdg','mathach','antgrdm','teachid','classpd']
            tokeep = [c for c in tokeep if c in eog.columns]
            math = eog.loc[eog.mastid.notnull() & eog.mathscal.notnull()
                        & (eog.administ.astype(str) == "0"), tokeep].rename(
                                columns={'teachid':'teachid_eogm','classpd':'classpd_eogm'})
            math = math.loc[~math.duplicated('mastid', keep=False)]

            # Find best reading
            tokeep = ['mastid','readscal','tchrdjdg','antgrdr','readach','teachid','classpd',]
            tokeep = [c for c in tokeep if c in eog.columns]
            read = eog.loc[eog.mastid.notnull() & eog.readscal.notnull()
                        & (eog.administ == "0"), tokeep].rename(
                                columns={'teachid':'teachid_eogr','classpd':'classpd_eogr'})
            read = read.loc[~read.duplicated('mastid', keep=False)]

            # Find skills stuff
            tokeep = ['mastid', 'homework', 'freeread', 'elctrnic', 'lookover', 'mathnotes', 'tv']   
            tokeep = [c for c in tokeep if c in eog.columns]
            if len(tokeep) > 0:
                noncog = eog.loc[eog.mastid.notnull(), tokeep]
                noncog = noncog.drop_duplicates('mastid', keep='first')
            else:
                noncog = None

            # Keep relevant vars
            eog = tmp.merge(math, how='left', on='mastid')
            eog = eog.merge(read, how='left', on='mastid')
            if noncog is not None:
                eog = eog.merge(noncog, how='left', on='mastid')

            # Add to lsit
            eogs += [eog,]

    # Put together
    eog = pd.concat(eogs, sort=False, ignore_index=True)

    # Standardize
    for col in ['mathscal','readscal']:
        eog[col] = (eog[col] - eog.groupby(['grade','year'])[col].transform('mean')
                        ) / eog.groupby(['grade','year'])[col].transform('std')

    # Reformat noncog variables
    eog['homework'] = pd.to_numeric(eog.homework.replace({'G':-1,'A':0,'B':1,'C':2,'D':3,'E':4,'F':5})) # time spent on homework   
    eog['freeread'] = pd.to_numeric(eog.freeread.replace({'A':0,'B':1,'C':2,'D':3,'E':4}))  # time spent free reading
    eog['tv'] = pd.to_numeric(eog.tv.replace({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}))    # time spent watching TV
    for col in ['homework','freeread','tv']:
        eog[col] = (eog[col] - eog.groupby(['grade','year'])[col].transform('mean')
                        ) / eog.groupby(['grade','year'])[col].transform('std')  
    eog.rename(columns={'tv':'watchtv'}, inplace=True)      

    # Save
    assert eog[['mastid','year','grade']].duplicated().sum() == 0
    eog.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/eog.pkl.gz", compression='gzip')


### 4) Add EOC data
if 0:
    print("\n\nAdding EOC files")
    eocs = []
    for year in range(firstyr-2,lastyr+1):
        tmp = []
        for subject in ['eng1','alg1','alg2','geom']:
            print("Loading EOC for {} subject {}".format(year,subject))
            try:
                with gzip.open(data_path + "/Student/End of Course/{}/{}pub{}.dta.gz".format(year,subject,str(year)[-2:]), 'rb') as inF:
                    eoc = pd.read_stata(inF)
            except Exception as e:
                print(e)
                continue

            # Find entry
            if 'administ' not in eoc.columns:
                eoc['administ'] = "0"
            if 'tjudge' in eoc.columns:
                eoc.rename(columns={'tjudge':'teachach'}, inplace=True)

            tokeep = ['mastid','antgrade'] + [c for c in eoc.columns if any([
                                a in c for a in ['scal','ach']]) 
                                and c not in ['teachid','comachom']] 

            eoc = eoc.loc[eoc.mastid.notnull() & eoc['{}scal'.format(subject)].notnull()
                    & (eoc.administ.astype(str) == "0"), tokeep]
            eoc = eoc.rename(columns={'antgrade':'{}antgrade'.format(subject)})
            if 'teachach' in eoc.columns:
                eoc = eoc.rename(columns={'teachach':'{}teachach'.format(subject)})

            # De-dup
            eoc = eoc.loc[~eoc.duplicated('mastid', keep=False)].set_index('mastid')
            tmp += [eoc,]

        # Add to lsit
        eoc = pd.concat(tmp, axis=1, sort=False, ignore_index=True)
        eoc['year'] = year
        eocs += [eoc,]

    # Put together
    eoc = pd.concat(eocs, sort=False)

    # Standardize
    for col in ['eng1scal','alg1scal','alg2scal','geomscal']:
        eoc[col] = (eoc[col] - eoc.groupby(['year'])[col].transform('mean')
                        ) / eoc.groupby(['year'])[col].transform('std')
    
    # Save
    eoc.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/eoc.pkl.gz", compression='gzip')


### 5) Add demographic data from various files
if 0:
    '''
    EOG coding changes:
    pared:
    1996-1999, 2001-2002:   1 = Did not finish high school
                            2 = High school graduate
                            3 = Trade or business school graduate
                            4 = Community, technical or junior college graduate
                            5 = Four-year college graduate
                            6 = Graduate school degree

    2003-2004:          1 = Did not finish high school
                        2 = High school graduate
                        3 = Some education after high school but did not graduate
                        4 = Community, technical or junior college graduate
                        5 = Trade or business school graduate
                        6 = Four-year college graduate
                        7 = Graduate school degree

    2000,2005-:         1 = Did not finish high school
                        2 = High school graduate
                        3 = Some education after high school but did not graduate
                        4 = Trade or business school graduate
                        5 = Community, technical or junior college graduate
                        6 = Four-year college graduate
                        7 = Graduate school degree

    except:
    1996-2000:    1 = Not identified as an Exceptional Student
                  2 = Academically/Intellectually Gifted (AIG)
                  3 = Behaviorally-Emotionally Handicapped
                  4 = Hearing Impaired
                  5 = Educable Mentally Handicapped
                  6 = Specific Learning Disabled
                  7 = Speech-Language Impaired
                  8 = Visually Impaired 
                  9 = Other Health Impaired
                10 = Orthopedically Impaired
                11 = Traumatic Brain Injured
                12 = Other Exceptional Classifications

    2001-2002:    1 = Not identified as an Exceptional Student
                  2 = Academically/Intellectually Gifted (AIG)
                  3 = Behaviorally-Emotionally Handicapped
                  4 = Hearing Impaired
                  5 = Educable Mentally Handicapped
                  6 = Specific Learning Disabled
                  7 = Speech-Language Impaired
                  8 = Visually Impaired 
                  9 = Other Health Impaired
                10 = Orthopedically Impaired
                11 = Traumatic Brain Injured
                12 = Autistic
                13 = Severe/Profound Mentally Disabled
                14 = Multihandicapped
                15 = Deaf-Blind
                16 = Trainable Mentally Handicapped

    2003-:        1 = Not identified as an Exceptional Student
                  2 = Academically/Intellectually Gifted (AIG)
                  3 = Behaviorally-Emotionally Handicapped
                  4 = Hearing Impaired
                  5 = Educable Mentally Handicapped
                  6 = Deaf-Blind
                  7 = Visually Impaired
                  8 = Other Health Impaired 
                  9 = Orthopedically Impaired
                10 = Traumatic Brain Injured
                11 = Severe/Profound Mentally Disabled 
                12 = Multihandicapped
                13 = Speech-Language Impaired
                14 = Trainable Mentally Handicapped
                15 = Specific Leaning Disabled
                16 = Autistic

    schllunch:
    1997-2002:  0 = Student not eligible for National School Lunch Program
                1 = Reduced price lunch
                2 = Free lunch
                3 = Information not available
                4 = School not participating

    2003-       2 = Free lunch
                3 = Reduced price lunch
                4 = Full pay; School not participating; Student not eligible

    
    EOC coding:
    pared:
    2003-:      A = Did not finish high school
                B = High school graduate
                C = Some additional education after high school, but did not graduate
                D = Trade or business school graduate
                E = Community, technical or junior college graduate
                F = Four-year college graduate
                G = Graduate school degree

    except:
    2003:        1 = Not identified as an Exceptional Student
                  2 = Academically/Intellectually Gifted (AIG)
                  3 = Behaviorally-Emotionally Handicapped
                  4 = Hearing Impaired
                  5 = Educable Mentally Handicapped
                  6 = Specific Learning Disabled
                  7 = Speech-Language Impaired
                  8 = Visually Impaired
                  9 = Other Health Impaired
                10 = Orthopedically Impaired
                11 = Traumatic Brain Injured
                12 = Autistic
                13 = Severe/Profound Mentally Disabled
                14 = Multihandicapped
                15 = Deaf-Blind
                16 = Trainable Mentally Handicapped

    2004-:      same as EOG

    MBuild coding:
    pared:
    2006-2008:      1 = Less than high school
                    2 = High school graduate
                    3 = Some education after high school
                    4 = Trade school
                    5 = Junior college 
                    6 = 4-year college
                    7 = Graduate school

    '''
    print("\n\nAdding demos from EOG / EOC / Mbuild files")
    demos = []
    for year in range(1996,lastyr+1):
        # EOG
        for grade in range(3,9):
            if ((year == 1996) & (grade >= 5)):
                continue
            print("Loading EOG for {} grade {}".format(year,grade))
            with gzip.open(data_path + "/Student/End of Grade/{}/eog{}pub{}.dta.gz".format(year,grade,str(year)[-2:]), 'rb') as inF:
                eog = pd.read_stata(inF)

            # Check if present
            cols = ['mastid','sex','ethnic','bdate','pared','except','id504','schlunch','frl','ed',
                        'limeng','lep','lang','lepstat','aigmath','aigread','twin']
            for col in cols:
                if col not in eog.columns:
                    eog[col] = None

            # Add to list
            eog = eog.loc[eog.mastid.notnull(), cols].drop_duplicates()
            eog['year'] = year
            eog['source'] = 'eog'
            demos += [eog,]

        # EOC
        for subject in ['eng1','alg1','alg2','ush','civ','geom','bio']:
            print("Loading EOC for {} subject {}".format(year,subject))
            try:
                with gzip.open(data_path + "/Student/End of Course/{}/{}pub{}.dta.gz".format(year,subject,str(year)[-2:]), 'rb') as inF:
                    eoc = pd.read_stata(inF)
            except Exception as e:
                print(e)
                continue

            # Check if present
            cols = ['mastid','sex','ethnic','pared','bdate','except','id504','schlunch','frl','lepstat',
                        'aigmath','aigread','twin']
            for col in cols:
                if col not in eoc.columns:
                    eoc[col] = None

            # Add to list
            eoc = eoc.loc[eoc.mastid.notnull(), cols].drop_duplicates()
            eoc['year'] = year
            eoc['source'] = 'eoc'

            demos += [eoc,]

        # Mbuild
        if year == 1996:
            continue        
        print("Loading Mbuild for {}".format(year))
        with gzip.open(data_path + "/Student/MBuild/mb_{}_pub.dta.gz".format(year), 'rb') as inF:
            mb = pd.read_stata(inF)

        # Check if present
        cols = ['mastid','sex','ethnic','bdate','pared','eds','frl',
                        'lep','lepflag','aig','aig_m','aig_math','aig_r']
        for col in cols:
            if col not in mb.columns:
                mb[col] = None

        # Add to list
        mb = mb.loc[mb.mastid.notnull(), cols].drop_duplicates()
        mb['year'] = year
        mb['source'] = 'mbuild'
        demos += [mb,]

    # Combine
    demo = pd.concat(demos, sort=False, ignore_index=True).drop_duplicates()

    # Recode twin indicators, which are year specific
    twin = demo.loc[demo.twin.notnull(), ['mastid','year','source','twin']].sort_values(['twin','mastid'])
    twin['twin_id'] = twin.groupby(['year','source','twin']).grouper.group_info[0]
    twin = twin.loc[twin.groupby('twin_id').mastid.transform('nunique') > 1]
    twin = twin.groupby('mastid').twin_id.max()

    # Recode parents education
    demo.loc[(demo.source == "eoc") & (demo.year < 2000), 'pared'] = None
    demo['pared_nohs'] = demo.pared.astype(str).isin(["1","A"]).astype(int)
    demo['pared_hsorless'] = demo.pared.astype(str).isin(["1","2","A","B"]).astype(int)
    demo['pared_somecol'] = demo.pared.astype(str).isin(["3","4","5","6","7","C","D","E","F","G"]).astype(int)
    demo['pared_baormore'] = demo.pared.astype(str).isin(["6","7","F","G"]).astype(int)
    demo.loc[demo.year.isin([1996,1997,1998,1999,2001,2002]) & (demo.source == 'eog'), 
                        'pared_baormore'] = demo.pared.astype(str).isin(["5","6"]).astype(int)
    for col in ['pared_nohs','pared_hsorless','pared_somecol','pared_baormore']:
        demo.loc[demo.pared.isnull() | (demo.pared == ""), col] = None

    # Recode exceptionality
    demo['exc_not'] = (pd.to_numeric(demo['except']) == 1).astype(int)
    demo['exc_aig'] = (pd.to_numeric(demo['except']) == 2).astype(int)
    demo['exc_behav'] = (pd.to_numeric(demo['except']) == 3).astype(int)
    demo['exc_educ'] = (pd.to_numeric(demo['except']) == 5).astype(int)
    for col in ['exc_not','exc_aig','exc_behav','exc_educ']:
        demo.loc[pd.to_numeric(demo['except']).isnull(), col] = None
    
    # id504 is sometimes 1 and sometimes "Y"
    demo.loc[demo.id504.notnull(), 'id504'] = demo.id504.isin(['1','Y',1]).astype(int)

    # Free lunches
    demo['fr_lunch'] = (demo.frl.isin(['F','R']) | demo.schlunch.isin([2,3,'2','3'])).astype(int)
    demo.loc[demo.schlunch.notnull() & (demo.year <= 2002), 'fr_lunch'] = demo.schlunch.isin(['1','2',1,2]).astype(int)
    demo.loc[demo.frl.isnull() & demo.schlunch.isnull(), 'fr_lunch'] = None

    # Gifted indicators
    demo.loc[demo.aigmath.notnull(), 'aigmath'] = demo.aigmath.isin(['1','Y',1]).astype(int)
    demo.loc[demo.aigread.notnull(), 'aigread'] = demo.aigread.isin(['1','Y',1]).astype(int)

    # Disadvantaged indicator
    demo.loc[demo.fr_lunch.notnull() | demo.eds.notnull(), 'disadv'] = ((demo.fr_lunch == 1) | (demo.eds == "Y")).astype(int)

    # LEP status
    demo['lim_eng'] = (demo.limeng.isin(['1',1,'Y']) | 
                        ((demo.source == "Mbuild") & demo.lep.isin(['Y','1',1])) |
                        demo.lepstat.isin([2,3,4,5,'2','3','4','5']) |
                        demo.lepflag.isin([1,2,3,4,'1','2','3','4'])
                        ).astype(int)
    demo.loc[demo.limeng.isnull() & demo.lep.isnull() & demo.lepstat.isnull() & demo.lepflag.isnull(), 'lim_eng'] = None

    # Make bdates datetime
    demo['bdate'] = pd.to_datetime(demo.bdate, errors='coerce')

    # Fix any collisions
    tmp = [twin,]
    for col in ['sex','ethnic','bdate']:
        print("Fixing collisions for {}".format(col))
        demo.loc[demo[col] == "", col] = None
        demo['{}_nuni'.format(col)] = demo.groupby(['mastid',col])['year'].transform('nunique')
        tmp += [demo.loc[demo['{}_nuni'.format(col)] == demo.groupby('mastid')['{}_nuni'.format(col)].transform('max'),
                ['mastid',col]].drop_duplicates('mastid').set_index('mastid'),]

    for col in ['pared_nohs','pared_hsorless','pared_somecol','pared_baormore']:
        print("Fixing collisions for {}".format(col))
        demo.loc[demo[col] == "", col] = None
        demo['{}_nuni'.format(col)] = demo.groupby(['mastid',col])['year'].transform('nunique')
        tmp += [demo.loc[demo['{}_nuni'.format(col)] == demo.groupby('mastid')['{}_nuni'.format(col)].transform('max'),
                ].groupby('mastid')[col].max(),]

    for col in ['exc_not','exc_aig','exc_behav','exc_educ','id504','fr_lunch','aigmath','aigread','disadv','lim_eng']:
        print("Taking max for {}".format(col))
        tmp += [demo.groupby('mastid')[col].max(),]
    
    tosave = pd.concat(tmp, axis=1, sort=False).reset_index()
    assert tosave.mastid.duplicated().sum() == 0

    # Save 
    tosave.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/demos.pkl.gz", compression='gzip')


### 6) Add disciplinary data
# Note: We can add indicators for: "disruptive", "aggressive", "violent" --- currently not included (look at Jon's code to see how to add this).
if 0:
    # list_susp = []
    # for year in list(np.arange(2001,2004+1)) + list(np.arange(2006,2013+1)):    
    #     print('Working on suspensions in year {}'.format(year), flush=True)

    #     # Load raw data
    #     if year == 2001:
    #         with gzip.open(data_path + "/Student/Suspension/mastsusp{}.dta.gz".format(str(year)[-2:]), 'rb') as inF:
    #             _tmp = pd.read_stata(inF, convert_dates = False)
    #     elif year in range(2002,2004+1):
    #         with gzip.open(data_path + "/Student/Suspension/mastsusp{}r.dta.gz".format(str(year)[-2:]), 'rb') as inF:
    #             _tmp = pd.read_stata(inF, convert_dates = False)
    #     elif year == 2005: #YST: 2005 does NOT have mastid... so cannot be linked...
    #         with gzip.open(data_path + "/Student/Suspension/susp05.dta.gz", 'rb') as inF:
    #             _tmp = pd.read_stata(inF, convert_dates = False)
    #     else:
    #         with gzip.open(data_path + "/Student/Suspension/mastsusp{}.dta.gz".format(year), 'rb') as inF:
    #             _tmp = pd.read_stata(inF, convert_dates = False)

    #     # Unify formats / variable names across years
    #     if year == 2001:
    #         _tmp['oss'] = _tmp['oss'].astype(int, errors = 'ignore')
             
    #         _tmp['ALP_enroll'] = (_tmp['alpprov']=='Y').astype(int)
    #         _tmp['inschsusp'] = (_tmp['discipl']==1).astype(int)
    #         _tmp['outschsusp'] = _tmp['discipl'].isin([2,3]).astype(int)
    #         _tmp['longoutschsusp'] = (_tmp['discipl'] == 4).astype(int) 
    #         _tmp['expell'] = (_tmp['discipl'] == 5).astype(int)
    #     elif year in range(2002,2005+1):
    #         if year < 2005:
    #             _tmp.rename(columns = {'c_ossdays':'oss', 'c_issdays':'iss'}, inplace=True)
    #             _tmp['inschsusp'] = (_tmp['c_iss']==1).astype(int)
    #         else:
    #             _tmp.rename(columns = {'c_ossdays':'oss'}, inplace=True)
    #         _tmp['altsch_enroll'] = (_tmp['c_altsch']==1).astype(int)
    #         _tmp['ALP_enroll'] = (_tmp['c_alp'] > 0).astype(int)
    #         _tmp['outschsusp'] = (_tmp['c_oss'] > 0).astype(int)
    #         _tmp['longoutschsusp'] = (_tmp['c_lts'] > 0).astype(int) 
    #         _tmp['expell'] = (_tmp['c_exp'] == 1).astype(int)
    #     elif year in range(2006,2007+1): 
    #         _tmp.rename(columns = {'c_sts':'c_iss'}, inplace=True)
    #         _tmp['oss'] = _tmp['c_stdays'] + _tmp['c_ltdays']
    #         _tmp['outschsusp'] = (_tmp['oss'] > 0).astype(int)
    #         _tmp['longoutschsusp'] = (_tmp['c_lts'] > 0).astype(int) 
    #         _tmp['expell'] = (_tmp['c_exp'] > 0).astype(int)
    #     elif year == 2008:
    #         _tmp['oss'] = _tmp['osstot'].astype(int, errors = 'ignore') 
    #         _tmp['expell'] = (_tmp['expsn'] == 1).astype(int)
    #         _tmp['outschsusp'] = (_tmp['oss'] > 0).astype(int) 
    #         _tmp['longoutschsusp'] = (_tmp['ltnum'] > 0).astype(int)
    #     elif year in range(2009,2013+1):
    #         _tmp['oss'] = _tmp['ossntot'].astype(int, errors = 'ignore')
    #         _tmp['expell'] = (_tmp['expsn'] == 1).astype(int)
    #         _tmp['outschsusp'] = (_tmp['oss'] > 0).astype(int) 
    #         _tmp['longoutschsusp'] = (_tmp['ltnum'] > 0).astype(int)

    #     if ('idate' in _tmp.columns):
    #         _tmp['offid'] = _tmp.groupby(['mastid','idate']).grouper.group_info[0]
    #     elif ('idate' in _tmp.columns):
    #         _tmp['offid'] = _tmp.groupby(['mastid','date']).grouper.group_info[0]
    #     else:
    #         _tmp['offid'] = _tmp['mastid'].copy()
    #     assert _tmp.offid.isnull().sum() == 0

    #     ### (0) Find variables that are available this year
    #     _tmp['numinfrac'] = 1
    #     key_vars = ['oss','numinfrac', 'ALP_enroll','inschsusp', 'outschsusp', 'longoutschsusp', 'expell','altsch_enroll']
    #     _vars = [l for l in _tmp.columns if l in key_vars]

    #     ### (1) Collapse by mastid-susid and take MAX
    #     _susp = _tmp.groupby(['mastid','offid'])[_vars].max().reset_index()

    #     ### (2) Collapse by mastid and take SUM
    #     _susp = _susp.groupby(['mastid'])[_vars].sum().reset_index()

    #     # Add to list
    #     _susp['year'] = year
    #     assert _susp.duplicated().sum() == 0
    #     assert _susp.mastid.duplicated().sum() == 0
    #     list_susp += [_susp,]

    # susp = pd.concat(list_susp)
    # susp.drop_duplicates(inplace=True)
    # assert susp[['mastid','year']].duplicated().sum() == 0
    # susp.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/susp.pkl.gz", compression='gzip')

    susps = []
    for year in range(2002,2017):
        if year == 2005:
            continue
        print('Working on suspensions in year {}'.format(year), flush=True)

        if year in [2002,2003,2004]:
            with gzip.open(data_path + "/Student/Suspension/mastsusp{}r.dta.gz".format(str(year)[-2:]), 'rb') as inF:
                _tmp = pd.read_stata(inF, convert_dates = False)
        else:
            with gzip.open(data_path + "/Student/Suspension/mastsusp{}.dta.gz".format(year), 'rb') as inF:
                _tmp = pd.read_stata(inF, convert_dates = False)
        
        consequence_columns = [c for c in _tmp.columns if 'c_conseq' in c or 'consqtype' in c]
        if year <= 2004:
            _tmp['any_discp'] = 1
            _tmp['iss'] = _tmp[consequence_columns].apply(lambda x: any([a == 3 for a in x]), axis=1).astype(int)
            _tmp['oss'] = _tmp[consequence_columns].apply(lambda x: any([a == 4 for a in x]), axis=1).astype(int)
            _tmp['explusion'] = _tmp[consequence_columns].apply(lambda x: any([a == 5 for a in x]), axis=1).astype(int)
            _tmp['detention'] = (_tmp.c_deten > 0).astype(int)          
        elif year >= 2006:
            _tmp['any_discp'] = 1
            _tmp['iss'] = _tmp[consequence_columns].apply(lambda x: any([(a == "002") | (a == "105") for a in x]), axis=1).astype(int)
            _tmp['oss'] = _tmp[consequence_columns].apply(lambda x: any([a in ["003","004","005","035"] for a in x]), axis=1).astype(int)
            _tmp['oss10'] = _tmp[consequence_columns].apply(lambda x: any([a in ["003","035"] for a in x]), axis=1).astype(int)
            _tmp['oss11'] = _tmp[consequence_columns].apply(lambda x: any([a == "004" for a in x]), axis=1).astype(int)
            _tmp['oss365'] = _tmp[consequence_columns].apply(lambda x: any([a == "005" for a in x]), axis=1).astype(int)
            _tmp['explusion'] = _tmp[consequence_columns].apply(lambda x: any([a == "006" for a in x]), axis=1).astype(int)
            _tmp['conf_parent'] = _tmp[consequence_columns].apply(lambda x: any([a == "030" for a in x]), axis=1).astype(int)
            _tmp['report_law'] = _tmp[consequence_columns].apply(lambda x: any([a == "012" for a in x]), axis=1).astype(int)
            _tmp['detention'] = _tmp[consequence_columns].apply(lambda x: any([a == "021" for a in x]), axis=1).astype(int)

        # Fix formats and keep students in grades 1-12
        _tmp['grade'] = pd.to_numeric(_tmp['grade'], errors='coerce')
        _tmp = _tmp.loc[_tmp.mastid.notnull() & _tmp.grade.notnull()]

        # Collapse to sums
        keepcols = ['any_discp','iss','oss','detention','oss10','oss11','oss365','explusion','conf_parent','report_law']
        s = _tmp.groupby('mastid')[[c for c in keepcols if c in _tmp.columns]].max().reset_index()
        s['year'] = year
        susps += [s,]

    susp = pd.concat(susps)
    susp.drop_duplicates(inplace=True)
    assert susp[['mastid','year']].duplicated().sum() == 0
    susp.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/susp.pkl.gz", compression='gzip')


### 6b) Add absences data from MBuild files
if 0:
    '''
    In mbuild, days daysabs exists only in 2004-2005m 2013-, days member exists in 2006- in mbuild files
    In accdemo, daysabs exists 2006-2015, also has times_tardy, unexc_abs, times_in_susp, times_out_susp, days_in_susp, days_out_susp
        at least in 2006-2011
    '''
    acc = []
    for yr in range(2004,2017):
        print("Loading attendence data for {}".format(yr))
        if yr <= 2005:
            with gzip.open(data_path + "/Student/MBuild/mb_{}_pub.dta.gz".format(yr), 'rb') as inF:
                _acc = pd.read_stata(inF)
        elif (yr>2005) & (yr <=2012):
            with gzip.open(data_path + "/Student/MBuild/ACCDEMO/accdemopub{}.dta.gz".format(yr), 'rb') as inF:
                _acc = pd.read_stata(inF)
        elif (yr > 2012) & (yr < 2016):
            with gzip.open(data_path + "/Student/MBuild/ACCDEMO/accdemo_pub{}.dta.gz".format(yr), 'rb') as inF:
                _acc = pd.read_stata(inF)
        elif yr == 2016:
            with gzip.open(data_path + "/Student/MBuild/pcaudit_pub{}.dta.gz".format(yr), 'rb') as inF:
                _acc = pd.read_stata(inF)

        # Fix formats and keep students in grades 1-12
        _acc['grade'] = pd.to_numeric(_acc['grade'], errors='coerce')
        _acc = _acc.loc[_acc.mastid.notnull() & _acc.grade.notnull()] 

        # Collect relevant information
        keepvars = ['daysabs', 'daysmem', 'times_tardy', 'days_in_susp', 'days_out_susp', 'times_in_susp', 'times_out_susp']
        _acc = _acc[[l for l in keepvars if l in _acc.columns] + ['mastid']]
        for k in _acc.columns:
            _acc[k] = pd.to_numeric(_acc[k], errors='coerce')
        for col in [l for l in keepvars if l in _acc.columns]:
            if _acc[col].isnull().min() == 1:
                _acc.drop(col, axis=1, inplace=True)
        _acc = _acc.groupby(['mastid'])[[l for l in keepvars if l in _acc.columns]].sum().reset_index()
        _acc['year'] = int(yr)
        acc += [_acc, ]

    acc = pd.concat(acc)
    acc.drop_duplicates(inplace=True)
    assert acc[['mastid','year']].duplicated().sum() == 0
    acc.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/absences.pkl.gz", compression='gzip')


### 7) Add graduation data
if 0:
    ls_files = []
    for yr in range(2006,2016+1):
        print('Working on Student Exit file from year {}'.format(yr))
        with gzip.open(data_path + "/Student/School Exit/exit_pub{}.dta.gz".format(str(yr)), 'rb') as inF:
            _tmp = pd.read_stata(inF, convert_dates = False)
        _tmp = _tmp.loc[_tmp.mastid.notnull(), ['mastid', 'effective_exit_code', 'grad_year', 'dropout_year']].drop_duplicates()

        _tmp = _tmp.loc[_tmp.effective_exit_code.isin(['4','6','9'])].drop_duplicates() # Keep if student dropped out, died, or graduated
        _tmp['grad']  = (_tmp['effective_exit_code'] == '9').astype(int)
        _tmp['dead']  = (_tmp['effective_exit_code'] == '6').astype(int)
        _tmp['dropout']  = (_tmp['effective_exit_code'] == '4').astype(int)

        for ll in ['grad', 'dead', 'dropout']:
            _tmp[ll] = _tmp.groupby(['mastid'])[ll].transform('max')
        _tmp['grad'].where((_tmp['dropout'] == 0) | (_tmp['dead'] == 0), 0, inplace=True)


        for ll in ['grad', 'dropout']:
            _tmp[ll+'_year'] = pd.to_numeric(_tmp[ll+'_year'], errors='coerce')
            _tmp[ll+'_year'].where(_tmp[ll] == 1, np.nan, inplace=True)
            _tmp[ll+'_year'] = _tmp.groupby(['mastid'])[ll+'_year'].transform('max')
        _tmp['year_stuExit_file'] = yr

        del _tmp['effective_exit_code']
        _tmp.drop_duplicates(inplace=True)

        assert _tmp.mastid.duplicated().sum() == 0
        ls_files += [_tmp,]
        del _tmp
    grd = pd.concat(ls_files)
    del ls_files

    # Fix multiple mastid enteries
    for ll in ['grad', 'dead', 'dropout']:
        grd[ll] = grd.groupby(['mastid'])[ll].transform('max')
    grd['grad'].where(grd['dead'] == 0, 0, inplace=True)
    grd['dropout'].where(grd['dead'] == 0, 0, inplace=True)

    for ll in ['grad', 'dropout']:
        grd[ll+'_year'] = grd.groupby(['mastid'])[ll+'_year'].transform('max')

    # Max year student appear's
    grd['max_year_stuExit_file'] = grd.groupby(['mastid'])['year_stuExit_file'].transform('max')
    del grd['year_stuExit_file']

    # Too little students die (about 1300) so dropping that variable
    del grd['dead']
    grd.drop_duplicates(inplace=True)

    # save
    assert grd.mastid.duplicated().sum() == 0
    # In [195]: grd.columns                                                                                                                                                      
    # Out[195]: Index(['mastid', 'grad_year', 'dropout_year', 'grad', 'dropout', 'max_year_stuExit_file'], dtype='object')
    grd[['mastid','grad','dropout']].to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/schoolExit.pkl.gz", compression='gzip')


### 9) Add 12th grade GPA data, college plans, etc.
if 0:
    print("\n\nWorking on GPA files")
    finals = []
    for year in range(2005,2011):
        print("Loading GPA data for {}".format(year))
        with gzip.open(data_path + "/Student/GPA/gpa{}.dta.gz".format(year), 'rb') as inF:
            gpa = pd.read_stata(inF)
            gpa['class_rank_uw'] = gpa.rank_unweighted / gpa.rank_number
            gpa['class_rank_w'] = gpa.rank_weighted / gpa.rank_number
            gpa = gpa[['mastid','bound_for_code','gpa_unweighted','gpa_weighted','class_rank_uw','class_rank_w']]
            gpa['year'] = year
        finals += [gpa,]

    for year in range(2012,2014):
        print("Loading GPA data for {}".format(year))
        with gzip.open(data_path + "/Student/GPA/gpa{}.dta.gz".format(year), 'rb') as inF:
            gpa = pd.read_stata(inF)
            gpa['class_rank_uw'] = gpa.rank_unweighted / gpa.rank_number
            gpa['class_rank_w'] = gpa.rank_weighted / gpa.rank_number
            gpa = gpa[['mastid','bound_for_code','gpa_unweighted','gpa_weighted','class_rank_uw','class_rank_w']]
            gpa['bound_for_code'] = gpa.bound_for_code.replace({'9':'1','10':'2','11':'3','12':'3','13':'5'})
            gpa['year'] = year
        finals += [gpa,]

    for year in range(2014,2017):
        print("Loading GPA data for {}".format(year))
        with gzip.open(data_path + "/Student/GPA/gpa{}.dta.gz".format(year), 'rb') as inF:
            gpa = pd.read_stata(inF)
            gpa['class_rank_uw'] = gpa.rank_unweighted / gpa.rank_number_unweight
            gpa['class_rank_w'] = gpa.rank_weighted / gpa.rank_number_weighted
            gpa = gpa[['mastid','bound_for','gpa_unweighted','gpa_weighted','class_rank_uw','class_rank_w']]
            gpa['bound_for_code'] = gpa.bound_for.replace({
                'COMMUNITY & TECHNICAL COLLEGES':'3',
                'PUBLIC SENIOR INSTITUTIONS':'1',
                'EMPLOYMENT':'7',                       
                'PRIVATE SENIOR INSTITUTIONS':'2',        
                'MILITARY':'6',                           
                'PUBLIC SENIOR INST -OUT OF NC':'1',      
                'PRIVATE SENIOR INST -OUT OF NC':'2',    
                'OTHER':'8',                            
                'TRADE, BUSINESS, NURSING SCHS':'5',       
                'COMM & TECH COLLEGES-OUT OF NC':'3',      
                'PRIVATE JUNIOR COLLEGE':'4',              
                'TRADE, BUS, NURS SCH-OUT OF NC':'5',
                'PRIVATE JR COLLEGE -OUT OF NC':'4',
                })
            gpa['year'] = year
        finals += [gpa.drop('bound_for',axis=1),]        
    
    # concat
    gpa = pd.concat(finals)
    gpa = gpa.drop_duplicates()
    gpa = gpa.loc[gpa.gpa_unweighted.notnull() | gpa.gpa_weighted.notnull()]
    gpa = gpa.loc[~gpa.duplicated(['mastid','year'], keep=False)]
    gpa = gpa.loc[gpa.year == gpa.groupby('mastid').year.transform('min')]

    # save
    assert gpa.mastid.duplicated().sum() == 0
    gpa.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/gpa.pkl.gz", compression='gzip')


### 8) Add school data
if 0:
    print("\n\nWorking on school files")
    finals = []
    for year in range(2006,2013+1):
        print("Loading school data for {}".format(year))
        with gzip.open(data_path + "/School/Public School Universe/ccdpsu{}.dta.gz".format(year), 'rb') as inF:
            schl = pd.read_stata(inF)
            schl['year'] = year

        schl = schl.rename(columns=lambda x: re.sub("[0-9]{2}$","",x))
        schl = schl[['lea','schlcode','chartr','magnet','type','year']].rename(columns={'type':'schl_type'})       # 1 = yes, 2 = no
        finals += [schl,]

    # Combine, collapse, and save
    schl = pd.concat(finals)
    schl['charter_schl'] = (schl.chartr.astype(str) == "1").astype(int)
    schl['magnet_schl'] = (schl.magnet.astype(str) == "1").astype(int)
    schl['voc_schl'] = (schl.schl_type.astype(str) == "3").astype(int)
    schl['spec_ed_schl'] = (schl.schl_type.astype(str) == "2").astype(int)
    schl['alt_schl'] = (schl.schl_type.astype(str) == "4").astype(int)
    schl = schl.groupby(['lea','schlcode'])[['charter_schl','magnet_schl','voc_schl','spec_ed_schl','alt_schl']].max().reset_index()
    schl.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/schls.pkl.gz", compression='gzip')


### 9) Add teacher Xs
if 0:
    print("\n\nWorking on teacher files")
    # Education
    finals = []
    for year in range(2006,2011+1):
        print("Loading teacher data for {}".format(year))
        with gzip.open(data_path + "/Teacher/Education/lsprseduc{}.dta.gz".format(year), 'rb') as inF:
            lic = pd.read_stata(inF)
            lic['year'] = year    
        finals += [lic,]

    for year in range(2012,2013+1):
        print("Loading teacher data for {}".format(year))
        with gzip.open(data_path + "/Teacher/Education/educ_pub{}.dta.gz".format(str(year)[-2:]), 'rb') as inF:
            lic = pd.read_stata(inF)
            lic['year'] = year    
        finals += [lic,]

    # Combine and collapse
    lic = pd.concat(finals)
    lic = lic.loc[lic.teachid.notnull()]
    lic.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/teach_educ.pkl.gz", compression='gzip')

    # License
    finals = []
    for year in range(2006,2011+1):
        print("Loading teacher data for {}".format(year))
        with gzip.open(data_path + "/Teacher/License/licsal_pay_lic{}.dta.gz".format(year), 'rb') as inF:
            lic = pd.read_stata(inF)
            lic['year'] = year    
        finals += [lic,]

    for year in range(2012,2013+1):
        print("Loading teacher data for {}".format(year))
        with gzip.open(data_path + "/Teacher/License/area_pub{}.dta.gz".format(str(year)[-2:]), 'rb') as inF:
            lic = pd.read_stata(inF).rename(columns={'cls_lvl_elvl_cd':'cls_lvl_elv_cd'})
            lic['year'] = year    
        finals += [lic,]

    # Combine and collapse
    lic = pd.concat(finals)
    lic = lic.loc[lic.teachid.notnull()]
    lic[['teachid','year','pgm_sts_cd','lic_type_cd','cls_lvl_cd','cls_lvl_elv_cd']].to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/teach_lic.pkl.gz", compression='gzip')
            # cls_lvl_elv_cd missing in 2010 / 2011
            # cls_lvl_cd changes in 2010 on...


### 10) SAR file data and student counts
if 0:
    print("\n\nWorking on SAR files")
    finals = []
    for year in range(1997,2014):
        print("Loading teacher data for {}".format(year))
        with gzip.open(data_path + "/Classroom/Personnel/pers{}.dta.gz".format(year), 'rb') as inF:
            sar = pd.read_stata(inF)
            sar['year'] = year    
        finals += [sar[['teachid','lea','schlcode','year','acadlvl','astype','subjct','crsnum','secnum','semstr']],]

    sar = pd.concat(finals, sort=False, ignore_index=True)
    sar['semstr'] = pd.to_numeric(sar.semstr)

    # # Add student counts
    # print("\n\nWorking on Student count files")
    # finals = []
    # for year in range(1997,2014):
    #     print("Loading student count data for {}".format(year))
    #     with gzip.open(data_path + "/Classroom/Student Count/studir{}.dta.gz".format(year), 'rb') as inF:
    #         scount = pd.read_stata(inF)
    #         scount['year'] = year    

    #         # Special course titles
    #         scount['special_title'] = scount.crstitle.str.contains("(EXCEPTIONAL|EX | EX$| EXC$| BEH$|EX. | NEEDS | SPECIAL N|AUTISTIC|EMOTIONAL|BEH |BEHAVIOR|BEHAVORIALLY|HANDICAPPED|HANDICAP|HAND\.|INDIVIDUALIZED|INDIVIDUALIZE|TRAINABLE|MENTALLY|DISABLED|SPECIAL|IMPAIRED|PROFOUND|IEP SKILLS|HEALTH|PHYSICAL| EMH )")

    #         # Student counts
    #         scount['gifted_share'] = scount.acadgft / scount.membcnt
    #         scount['speced_share'] = (scount.autistic + scount.bemh + scount.dfblnd + scount.lrngdis + scount.hrgimp) / scount.membcnt

    #         finals += [scount[['year','lea','schlcode','crsnum','secnum','semstr','special_title','gifted_share','speced_share']]]

    # scounts = pd.concat(finals, sort=False, ignore_index=True)
    # scounts['semstr'] = pd.to_numeric(scounts.semstr)
    # scounts.drop_duplicates(inplace=True)
    # assert scounts.duplicated(['year','lea','schlcode','crsnum','secnum','semstr']).sum()

    sar.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/sar.pkl.gz", compression='gzip')


### 10) Build crime indicators
if 1:
    # Latest AOC data (dates are basically through the end of 2019)
    aoc_data = pd.read_pickle("/scratch/public/wacrim/aoc_data/offense_newonly_2020-09-08.pkl", compression='gzip')

    # Remove motions --- rows with a charge code of "MOTIONS"
    aoc_data = aoc_data.loc[~aoc_data.off_charged_code.isin([5046])]   

    # Remove all habitual charges and other enhancements, 9901 is extradition, 9902 Habeas, 9904 is governors warrant
    aoc_data = aoc_data.loc[~aoc_data.off_charged_code.isin([9922,9923,9921,2316,5256,5527,1393,9901,9902,9904])]  

    # Drop show cause failures to comply, contempts, FTAs, bill of particulars
    aoc_data = aoc_data.loc[~aoc_data.off_charged_code.isin([5034,5022,5026,5028,5029,5020])]  

    # Add sentence data
    aoc_data.loc[aoc_data.min_sent_frame == "D", 'sent'] = pd.to_numeric(aoc_data.min_sent_length, errors='coerce')    
    aoc_data.loc[aoc_data.min_sent_frame == "M", 'sent'] = pd.to_numeric(aoc_data.min_sent_length, errors='coerce') * 30.42        
    aoc_data.loc[aoc_data.min_sent_frame == "Y", 'sent'] = pd.to_numeric(aoc_data.min_sent_length, errors='coerce') * 365.25   

    # 09/2020 pii and data
    if 1:
        ids = pd.read_pickle(home + '/data/Crime/match_09_2020/NC_ids_crosswalk_2020-09-08.pkl.gz', compression='gzip')
    # 07/2017 pii and data
    else:
        ids = pd.read_pickle(home + '/data/Crime/match_05_2017/person_identifiers_NCERDC.pkl.gz', compression='gzip').drop(['defssn','sbi_no','fbi_no','initial_group3','initial_group8','defadd_street','defadd_city','defadd_state','defadd_zip'], axis=1)

    aoc_data = aoc_data.merge(ids[['crrkey','group_id']], how='left', on='crrkey')

    # Load codes and categories
    codes = pd.read_csv(home + "/teachers_final/aux/valid_codes.csv")
    ucr_codes = pd.read_csv(home + "/teachers_final/aux/nc_offensecodes_categories.csv")

    # Any arrest, any valid arrest, ucr categories, same for conviction
    aoc_data['any'] = 1
    aoc_data['crim'] = aoc_data.off_arrg_code.isin(codes.loc[codes.Exclude != "Yes"].CODE.unique()).astype(int)
    aoc_data['index'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['UCR Code Classification'].str.strip().isin([
                            'Aggravated Assault','Forcible Rape','Robbery','Criminal Homicide','Arson','Burglary','Larceny','Motor Vehicle Theft']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['viol'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Aggravated Assault','Other Assaults','Criminal Homicide','Kidnapping','Forcible Rape','Robbery']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['dwi'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Driving Under the Influence']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['drug'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Drug Abuse Violations']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['traff'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Traffic Offenses']),
                                'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['infrac'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                            'Infractions']),
                                'NCAOC Criminal Code'].unique()).astype(int)

    # Additional crime categories added by YST to calculate costs of crime measures (namings are to match READI QJI Table Table A.XVIII in WP)
    aoc_data['cost_murder'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Criminal Homicide']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_rape'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Forcible Rape','Sex Offenses']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_roberry'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Robbery']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_aggravated_assault'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Aggravated Assault','Kidnapping','Abuse of Elderly/Blind/Disabled','Explosives, Terrorism, and Bombing Offenses']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_assault'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Other Assaults','Offenses Against the Family and Children']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_burglary'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Burglary']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_motor_vehicle_theft'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Motor Vehicle Theft']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_larceny'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Larceny/Theft (except motor vehicle theft)','Stolen Property: Buying, Receiving, Possessing']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_drunk_driving_crash'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Driving Under the Influence']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_arson'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Arson']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_vandalism'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Damage Property/Vandalism']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_fraud'] = aoc_data.off_arrg_code.isin(ucr_codes.loc[ucr_codes['Case Category for SEP Outcome Study'].str.strip().isin([
                        'Fraud','Forgery and Counterfeiting']),
                            'NCAOC Criminal Code'].unique()).astype(int)
    aoc_data['cost_other_offense'] = 1 -  aoc_data.filter(regex=("cost_.*")).max(axis=1)
    assert aoc_data.filter(regex=("cost_.*")).max(axis=1).min() == 1

    # We inflate the cost estimate for each crime type by the likelihood of reporting and then by the likelihood an arrest will be made 
    aoc_data['cost_wtp'] = (aoc_data['cost_murder']/1/0.37)*13429758 + (aoc_data['cost_rape']/0.28/0.32)*327705 + (aoc_data['cost_roberry']/0.09/0.52)*304179 + (aoc_data['cost_aggravated_assault']/0.15/0.56)*76961 + (aoc_data['cost_assault']/0.11/0.37)*15014  + (aoc_data['cost_burglary']/0.09/0.48)*37476 + (aoc_data['cost_motor_vehicle_theft']/0.12/0.78)*15487 + (aoc_data['cost_larceny']/0.11/0.28)*3310 + (aoc_data['cost_drunk_driving_crash']/0.09/1)*68095 + (aoc_data['cost_arson']/0.09/1)*133115 + (aoc_data['cost_vandalism']/0.09/0.28)*1182 + (aoc_data['cost_fraud']/0.09/0.28)*4138 + (aoc_data['cost_other_offense']/0.09/0.28)*591
    aoc_data['cost_bottomup'] = (aoc_data['cost_murder']/1/0.37)*5910985 + (aoc_data['cost_rape']/0.28/0.32)*177330 + (aoc_data['cost_roberry']/0.09/0.52)*59110 +  (aoc_data['cost_aggravated_assault']/0.15/0.56)*65021 + (aoc_data['cost_assault']/0.11/0.37)*13004 + (aoc_data['cost_burglary']/0.09/0.48)*5911 + (aoc_data['cost_motor_vehicle_theft']/0.12/0.78)*10640 + (aoc_data['cost_larceny']/0.11/0.28)*1892 + (aoc_data['cost_drunk_driving_crash']/0.09/1)*35466 + (aoc_data['cost_arson']/0.09/1)*70932 + (aoc_data['cost_vandalism']/0.09/0.28)*1620 + (aoc_data['cost_fraud']/0.09/0.28)*3665 + (aoc_data['cost_other_offense']/0.09/0.28)*591
    aoc_data = aoc_data.drop(['cost_murder','cost_rape','cost_roberry','cost_aggravated_assault','cost_assault','cost_burglary','cost_motor_vehicle_theft','cost_larceny','cost_drunk_driving_crash','cost_arson','cost_vandalism','cost_fraud','cost_other_offense'], axis=1)
    assert ((aoc_data['cost_bottomup']<=0).sum()==0) & (aoc_data['cost_wtp']<=0).sum()==0

    # Add convictions
    cols = ['any','crim','index','viol','dwi','drug','traff','infrac']
    for col in cols:
        aoc_data[col + '_conv'] = ((aoc_data[col] == 1) & (aoc_data.off_convic_code.notnull())).astype(int)

    # Add incar
    aoc_data['incar'] = aoc_data.sent.notnull().astype(int)

    # Subset 
    tosave = aoc_data[['group_id','off_date','incar'] + cols + [c + '_conv' for c in cols] + ['cost_wtp','cost_bottomup']]
    tosave = tosave.loc[tosave.group_id.notnull() & tosave.off_date.notnull()]
    tosave = tosave.drop_duplicates()   # This removes multiple charges for same categories stemming from offenses on the same day

    # Replace identifiers
    ident = pd.read_sas(home + "/data/Crime/match_09_2020/ncindent20_match.sas7bdat", encoding='ascii').rename(columns={'person_id':'group_id'})
    tosave = tosave.merge(ident[['group_id','mastid','matchtype']], how='inner', on='group_id')
    tosave = tosave.loc[tosave.mastid.notnull()]

    # Save
    tosave.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/aoc.pkl.gz", compression='gzip')

    # # # Load up the matched data and compare
    # matches = pd.read_pickle(home + '/data/Crime/offense_dates_aoc_Aug2019.pkl.gz_repl.pkl.gz', compression='gzip')
    # test = np.random.choice(matches.mastid)
    # print(matches.loc[matches.mastid == test])
    # print(tosave.loc[tosave.mastid == test])
    # tmp = tosave.groupby('mastid').off_date.min().to_frame()
    # tmpold = matches.groupby('mastid').off_date_any16to20.min().to_frame()
    # tmpold = tmpold.merge(tmp, how='left', left_index=True, right_index=True)
    # tmpold['match'] = tmpold.off_date_any16to20 == tmpold.off_date
    # tmpold.loc[tmpold.off_date_any16to20.notnull()].match.value_counts(normalize=True)
    # tmpold.loc[tmpold.off_date_any16to20.notnull()].off_date.isnull().value_counts(normalize=True)

### 11) Combine into panel pased on EOG data
if 1:
    # Load EOG files
    data = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/eog.pkl.gz", compression='gzip').drop(['sex','ethnic','bdate'], axis=1).drop_duplicates()
    data = data.loc[~data.duplicated(['mastid','year'])]
        # Unique in mastid-year, drops ssmall share of sstudents in multipel schools

    # Add teacher assignments from CMB files
    cmb = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/cmb.pkl.gz", compression='gzip').rename(
            columns={'sex':'sex_cmb','ethnic':'ethnic_cmb'})
    cmb = cmb.loc[cmb.grade.between(3,8)]
    cmb = cmb.loc[cmb.teachid.notnull()]
    for course, lb, ub in [('hr',0,0),('math',2000,2999),('eng',1000,1999)]:
        tmp = cmb.loc[cmb.coursecode.between(lb,ub), ['mastid','year','grade','teachid','classroom_id','semester']].drop_duplicates()
        tmp['semester'] = pd.to_numeric(tmp.semester, errors='coerce').fillna(0)
        tmp = tmp.loc[tmp.semester == tmp.groupby(['mastid','year','grade']).semester.transform('max')] # Take spring semester where not null
        tmp = tmp.loc[tmp.groupby(['mastid','year','grade']).teachid.transform('nunique') == 1]
        tmp = tmp.groupby(['mastid','year','grade'])[['teachid','classroom_id']].first().reset_index()
        tmp = tmp.rename(columns={'teachid':'cmb_teachid_{}'.format(course),'classroom_id':'cmb_classroom_id_{}'.format(course)})
        data = data.merge(tmp, how='left', on=['mastid','year','grade'])

    # Add teacher validity info from SAR files to each teacher ID
    sar = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/sar.pkl.gz", compression='gzip')
    sar['sar_reg_hr'] = ((pd.to_numeric(sar.subjct, errors='coerce') == 0)*(sar['astype'] == "TE")*(pd.to_numeric(sar.acadlvl, errors='coerce') == 2)).astype(int)
    sar['sar_reg_math'] = ((pd.to_numeric(sar.subjct, errors='coerce').between(2000,2999))*(sar['astype'] == "TE")*(pd.to_numeric(sar.acadlvl, errors='coerce') == 2)).astype(int)
    sar['sar_reg_eng'] = ((pd.to_numeric(sar.subjct, errors='coerce').between(1000,1999))*(sar['astype'] == "TE")*(pd.to_numeric(sar.acadlvl, errors='coerce') == 2)).astype(int)
    for col in ['teachid_eogm','teachid_eogr','cmb_teachid_hr','cmb_teachid_math','cmb_teachid_eng']:
        sar[col] = sar.teachid
        sar['{}_sar_valid_hr'.format(col)] = sar.sar_reg_hr
        sar['{}_sar_valid_math'.format(col)] = sar.sar_reg_math
        sar['{}_sar_valid_eng'.format(col)] = sar.sar_reg_eng
        data = data.merge(sar.groupby([col,'lea','schlcode'])[['{}_sar_valid_hr'.format(col),'{}_sar_valid_math'.format(col),'{}_sar_valid_eng'.format(col)]].max().reset_index(), 
                    how='left', on=[col,'lea','schlcode']) # YST: changed here [] to [[]] on June 14, 2024 to match pandas updated version syntex

    ## Prioritize CMB matches
    # Define teacher indicators for homeroom teachers
    data['teachid_hr'] = data.cmb_teachid_hr.where((data.grade <= 5) & (data.cmb_teachid_hr_sar_valid_hr == 1), np.NaN)  
    data.loc[data.teachid_hr.isnull() & (data.grade <= 5) & (data.teachid_eogr_sar_valid_hr == 1), 'teachid_hr'] = data.teachid_eogr
    data.loc[data.teachid_hr.isnull() & (data.grade <= 5) & (data.teachid_eogm_sar_valid_hr == 1), 'teachid_hr'] = data.teachid_eogm
    print(data.groupby(['year','grade']).teachid_hr.apply(lambda x: x.isnull().mean()).unstack())
    print(data.loc[(data.grade <= 5) & data.year.between(1997,2013)].teachid_hr.isnull().mean())

    # Define teacher indicators for math
    data['teachid_math'] = data.cmb_teachid_math.where((data.cmb_teachid_math_sar_valid_math == 1), np.NaN)  
    data.loc[data.teachid_math.isnull() & (data.teachid_eogm_sar_valid_math == 1), 'teachid_math'] = data.teachid_eogm
    data.loc[data.teachid_math.isnull() & (data.teachid_eogr_sar_valid_math == 1), 'teachid_math'] = data.teachid_eogr
    print(data.groupby(['year','grade']).teachid_math.apply(lambda x: x.isnull().mean()).unstack())
    print(data.teachid_math.isnull().mean())

    # Define teacher indicators for eng
    data['teachid_eng'] = data.cmb_teachid_eng.where((data.cmb_teachid_eng_sar_valid_eng == 1), np.NaN)  
    data.loc[data.teachid_eng.isnull() & (data.teachid_eogr_sar_valid_eng == 1), 'teachid_eng'] = data.teachid_eogr
    data.loc[data.teachid_eng.isnull() & (data.teachid_eogm_sar_valid_eng == 1), 'teachid_eng'] = data.teachid_eogm
    print(data.groupby(['year','grade']).teachid_eng.apply(lambda x: x.isnull().mean()).unstack())
    print(data.teachid_eng.isnull().mean())

    # data['teachid_hr'] = data.teachid_eogm.where((data.grade <= 5) & (data.teachid_eogm_sar_valid_hr == 1), np.NaN)  
    # data.loc[data.teachid_hr.isnull() & (data.grade <= 5) & (data.teachid_eogr_sar_valid_hr == 1), 'teachid_hr'] = data.teachid_eogr
    # data.loc[data.teachid_hr.isnull() & (data.grade <= 5), 'teachid_hr'] = data.cmb_teachid_hr
    # print(data.groupby(['year','grade']).teachid_hr.apply(lambda x: x.isnull().mean()).unstack())
    # print(data.loc[(data.grade <= 5) & data.year.between(1997,2013)].teachid_hr.isnull().mean())

    # # Define teacher indicators for math
    # data['teachid_math'] = data.teachid_eogm.where((data.teachid_eogm_sar_valid_math == 1), np.NaN)  
    # data.loc[data.teachid_math.isnull() & (data.teachid_eogr_sar_valid_math == 1), 'teachid_math'] = data.teachid_eogr
    # data.loc[data.teachid_math.isnull(), 'teachid_math'] = data.cmb_teachid_math
    # print(data.groupby(['year','grade']).teachid_math.apply(lambda x: x.isnull().mean()).unstack())
    # print(data.teachid_math.isnull().mean())

    # # Define teacher indicators for eng
    # data['teachid_eng'] = data.teachid_eogm.where((data.teachid_eogm_sar_valid_eng == 1), np.NaN)  
    # data.loc[data.teachid_eng.isnull() & (data.teachid_eogr_sar_valid_eng == 1), 'teachid_eng'] = data.teachid_eogr
    # data.loc[data.teachid_eng.isnull(), 'teachid_eng'] = data.cmb_teachid_eng
    # print(data.groupby(['year','grade']).teachid_eng.apply(lambda x: x.isnull().mean()).unstack())
    # print(data.teachid_eng.isnull().mean())

    # Non-cognitive stuff from EOG
    for col in ['homework','freeread','watchtv']:
        data['_tmp'] = data[col].notnull()
        data['_validSGY'] = data.groupby(['year', 'schlcode', 'lea'])['_tmp'].transform('mean') > 0.75
        data.loc[~data._validSGY, col] = np.NaN

    # Ensure valid teachers
    for col in ['teachid_hr','teachid_eng','teachid_math']:
        data.loc[~data[col].apply(lambda x: len(str(int(x))) if pd.notnull(x) else 0).between(6,7), col] = np.NaN

    # Add classroom indicators
        # TBC

    # Drop extraneous columns
    todrop = [c for c in data.columns if 'sar_valid' in c]
    data = data.drop(todrop, axis=1)

    # Add schools
    schl = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/schls.pkl.gz", compression='gzip')
    data = data.merge(schl, how='left', on=['lea','schlcode'])

    # Add other data
    demos = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/demos.pkl.gz", compression='gzip')
    susp = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/susp.pkl.gz", compression='gzip')
    absences = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/absences.pkl.gz", compression='gzip')
    grd = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/schoolExit.pkl.gz", compression='gzip')
    gpa = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/gpa.pkl.gz", compression='gzip')
    data = data.merge(demos, how='left', on='mastid')
    data = data.merge(susp, how='left', on=['mastid','year'])
    data = data.merge(absences, how='left', on=['mastid','year'])
    data = data.merge(grd, how='left', on='mastid')
    data = data.merge(gpa.drop('year', axis=1), how='left', on='mastid')

    # Leads and lags of suspensions and absences
    for df in [susp[['mastid','year','any_discp','iss','oss','detention']], absences[['mastid','year','daysabs']]]:
        for label, transform in [('lead4_',-4),('lead3_',-3),('lead2_',-2),('lead1_',-1),('lag1_',1)]:
            tmp = df.copy()
            tmp['year'] = tmp.year + transform
            tmp = tmp.set_index(['year','mastid'])
            tmp.columns = [label + c for c in tmp.columns]
            data = data.merge(tmp.reset_index(), how='left', on=['mastid','year'])

    # Set suspension data to missing in school-years where there are zero records
        # 2007 through 2016, 
    for var in ["","lead1_","lead2_","lead3_","lead4_","lag1_"]:
        # data['_tmp'] = data[var + 'any_discp'].notnull()
        # data['_validSGY'] = data.groupby(['year', 'schlcode', 'lea'])['_tmp'].transform('sum') > 0
        if var == "":
            adj = 0
        elif "lead" in var:
            adj = int(var[4])
        elif "lag" in var:
            adj = -int(var[3])
        data['_validSGY'] = (data.year + adj).between(2007,2016)
        for k in [var + l for l in susp.columns if l not in ['mastid', 'year']]:
            if k in data.columns:
                data[k] = data[k].fillna(0)
                data.loc[~data._validSGY, k] = np.NaN
    # data.drop(['_validSGY', '_tmp'], axis=1, inplace = True)     
    data.drop(['_validSGY'], axis=1, inplace = True)     

    # Fix missing years in gpa data
    data.loc[~((12-data.grade)+data.year).between(2005,2016) | ((12-data.grade)+data.year == 2011),
                ['gpa_unweighted','gpa_weighted','class_rank_uw','class_rank_w']] = np.NaN
    print(data.groupby(['year','grade']).gpa_unweighted.apply(lambda x: x.notnull().mean()).unstack())

    # Fix missing years in absences data | We have this from 2004 through 2016
    for var in ["","lead1_","lead2_","lead3_","lead4_","lag1_"]:
        # data['_tmp'] = data[var + 'daysabs'].notnull()
        # data['_validSGY'] = data.groupby(['year', 'schlcode', 'lea'])['_tmp'].transform('sum') > 0
        if var == "":
            adj = 0
        elif "lead" in var:
            adj = int(var[4])
        elif "lag" in var:
            adj = -int(var[3])
        data['_validSGY'] = (data.year + adj).between(2004,2016)
        data[var + 'daysabs'] = data[var + 'daysabs'].fillna(0)
        data.loc[~data._validSGY, var + 'daysabs'] = np.NaN
    # data.drop(['_validSGY', '_tmp'], axis=1, inplace = True)     
    data.drop(['_validSGY'], axis=1, inplace = True)     

    # Fix missing years in grad data | Observed from 2007-2016, 
    data.loc[~(12-data.grade + data.year).between(2007,2016), 'grad'] = np.NaN

    # Code sex / ethnicity
    data['female'] = (data['sex'] == 'F').astype(int)
    data['female'].where(data['sex'].notnull(), np.nan, inplace=True)
    data['black'] = (data['ethnic'] == 'B').astype(int)
    data['white'] = (data['ethnic'] == 'W').astype(int)
    data['hispanic'] = (data['ethnic'] == 'H').astype(int)
    for ll in ['black', 'white', 'hispanic']:
        data[ll].where(data['ethnic'].notnull(), np.nan, inplace=True)

    # Add lag reading / math scores and study skills
    eog = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/eog.pkl.gz", compression='gzip').drop(['sex','ethnic','bdate'], axis=1).drop_duplicates()
    eog['year_orig'] = eog.year
    eog['year'] = eog.year + 1
    eog = eog.loc[~eog.duplicated(['mastid','year'], keep=False)]       # about 800 obs. Can only happen if a mastid appears in multiple grades in the same year.
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lag1_mathscal','readscal':'lag1_readscal',
                            'watchtv':'lag1_watchtv','homework':'lag1_homework','freeread':'lag1_freeread'}),
                    how='left', on=['mastid','year'])

    eog['year'] = eog.year_orig + 2
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lag2_mathscal','readscal':'lag2_readscal',
                        'watchtv':'lag2_watchtv','homework':'lag2_homework','freeread':'lag2_freeread'}),
                    how='left', on=['mastid','year'])
    eog['year'] = eog.year_orig

    # Add leads of math / reading scoores
    eog['year'] = eog.year_orig - 1
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lead1_mathscal','readscal':'lead1_readscal',
                            'watchtv':'lead1_watchtv','homework':'lead1_homework','freeread':'lead1_freeread'}),
                    how='left', on=['mastid','year'])
    eog['year'] = eog.year_orig - 2
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lead2_mathscal','readscal':'lead2_readscal',
                            'watchtv':'lead2_watchtv','homework':'lead2_homework','freeread':'lead2_freeread'}),
                    how='left', on=['mastid','year'])
    eog['year'] = eog.year_orig - 3
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lead3_mathscal','readscal':'lead3_readscal',
                            'watchtv':'lead3_watchtv','homework':'lead3_homework','freeread':'lead3_freeread'}),
                    how='left', on=['mastid','year'])
    eog['year'] = eog.year_orig - 4
    data = data.merge(eog[['mastid','year','mathscal','readscal','watchtv','homework','freeread']].rename(
                    columns={'mathscal':'lead4_mathscal','readscal':'lead4_readscal',
                            'watchtv':'lead4_watchtv','homework':'lead4_homework','freeread':'lead4_freeread'}),
                    how='left', on=['mastid','year'])

    # Remove leads where they shouldn't exist
    for lead in range(1,4):
        for col in ['mathscal','readscal','watchtv','homework','freeread']:
            data.loc[data.grade >= 8-(lead-1), 'lead{}_{}'.format(lead,col)] = np.NaN

    # Lag grade indicators
    tmp = data.groupby(['mastid','year']).grade.max().reset_index()
    tmp.rename(columns={'grade':'lag_grade'}, inplace=True)
    tmp['year'] = tmp.year + 1
    data = data.merge(tmp, how='left', on=['mastid','year'])

    # Grade repeitition indiciator
    data['grade_rep'] = (data.grade == data.lag_grade).astype(int)

    # Propspective grade repetition
    tmp = data.groupby(['mastid','year']).grade.max().reset_index()
    tmp.rename(columns={'grade':'lead_grade'}, inplace=True)
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp, how='left', on=['mastid','year'])
    data['lead_grade_rep'] = (data.grade == data.lead_grade).astype(int)

    # Leads of grade repetition
    tmp = data.groupby(['mastid','year']).lead_grade_rep.max().reset_index()
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp.rename(columns={'lead_grade_rep':'lead1_grade_rep'}), how='left', on=['mastid','year'])
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp.rename(columns={'lead_grade_rep':'lead2_grade_rep'}), how='left', on=['mastid','year'])
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp.rename(columns={'lead_grade_rep':'lead3_grade_rep'}), how='left', on=['mastid','year'])
    tmp['year'] = tmp.year - 1
    data = data.merge(tmp.rename(columns={'lead_grade_rep':'lead4_grade_rep'}), how='left', on=['mastid','year'])

    # Age at end of 2015, 2019
    data['age2005'] = (pd.to_datetime("2005-12-31") - pd.to_datetime(data.bdate, format="%Y-%m-%d")).dt.days / 365.25 
    data['age2015'] = (pd.to_datetime("2015-12-31") - pd.to_datetime(data.bdate, format="%Y-%m-%d")).dt.days / 365.25 
    data['age2019'] = (pd.to_datetime("2019-12-31") - pd.to_datetime(data.bdate, format="%Y-%m-%d")).dt.days / 365.25 

    # # Add incarceration age
    # scomp = pd.read_pickle('/accounts/projects/crwalters/cncrime/data/Crime/NCERDC_of_sentence_component_April_1_2019.pkl.gz_repl.pkl.gz')
    # tmp = scomp.loc[pd.to_numeric(scomp.CMPREFIX, errors='coerce').isnull()].groupby('mastid').conv_date.min().reset_index()
    # tmp = tmp.loc[tmp.conv_date.notnull()].rename(columns={'conv_date':'incar_age'})
    # data = data.merge(tmp, how='left', on='mastid')
    # data['incar_age'] = (data.incar_age - pd.to_datetime(data.bdate, format="%Y-%m-%d")).dt.days / 365.25 

    # # Add any interaction with CJ age
    # tmp = scomp.groupby('mastid').min_off_commit_date.min().reset_index()
    # tmp = tmp.loc[tmp.min_off_commit_date.notnull()].rename(columns={'min_off_commit_date':'anysc_age'})
    # data = data.merge(tmp, how='left', on='mastid')
    # data['anysc_age'] = (data.anysc_age - pd.to_datetime(data.bdate, format="%Y-%m-%d")).dt.days / 365.25

    # # interaction and incar between 16-21
    # data['cansee_sc1621'] = (data.age2015 >= 22).astype(int)
    # data['cansee_sc1621'] = (data.groupby(['grade','year']).cansee_sc1621.transform('mean') >= 0.9).astype(int)
    # data.loc[data.cansee_sc1621 == 1, 'anysc_1621'] = data.anysc_age.fillna(0).between(16,22).astype(int)
    # data.loc[data.cansee_sc1621 == 1, 'incar_1621'] = data.incar_age.fillna(0).between(16,22).astype(int)

    # New AOC data
    aoc = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/aoc.pkl.gz", compression='gzip')
    aoc = aoc.merge(data.groupby('mastid').bdate.min().reset_index(), how='inner', on='mastid')

    # Offenses age 16-21
    aoc['age'] = (aoc.off_date - pd.to_datetime(aoc.bdate, format="%Y-%m-%d")).dt.days / 365.25
    aoc = aoc.loc[aoc.age.between(16,21)]

    # Add to frame
    cols = ['incar','any','crim','index','viol','dwi','drug','traff','infrac']
    cols += [c + '_conv' for c in cols if c not in ['incar']] + ['cost_wtp','cost_bottomup']
    data = data.merge(aoc.groupby('mastid')[cols].max().rename(columns=lambda x: 'aoc_' + x).reset_index(), how='left', on='mastid')
    for col in ['aoc_' + c for c in cols]:
        data[col] = data[col].fillna(0)

    # YST: Added on June 19, 2024
    data = data.merge(aoc.groupby('mastid')[cols].sum().rename(columns=lambda x: 'aoc_total_' + x).reset_index(), how='left', on='mastid')
    for col in ['aoc_total_' + c for c in cols]:
        data[col] = data[col].fillna(0)

    # Set to missing where required
    data['cansee_aoc1621'] = ((data.age2019 >= 22) & (data.age2005 <= 16)).astype(int)
    data['cansee_aoc1621'] = (data.groupby(['grade','year']).cansee_aoc1621.transform('mean') >= 0.9).astype(int)
    data.loc[data.cansee_aoc1621 == 0, ['aoc_' + c for c in cols] + ['aoc_total_' + c for c in cols]] = np.NaN # YST: added "+ ['aoc_total_' + c for c in cols]"
    print(data.groupby(['year','grade']).aoc_any.apply(lambda x: x.notnull().mean()).unstack())
    print(data.groupby(['year','grade']).aoc_any.mean().unstack())

    # # Add any AOC age
    # aoc = pd.read_stata('/accounts/projects/crwalters/cncrime/users/wmorrison/data/crime_August_27_2019.dta')
    # aoc = aoc.loc[aoc.off_date_any16to25.notnull()].rename(columns={'off_date_any16to25':'aoc_age'})
    # aoc = aoc.groupby('mastid').aoc_age.min().reset_index()
    # data = data.merge(aoc, how='left', on='mastid')
    # data['anyaoc_age'] = (data.aoc_age - pd.to_datetime(data.bdate, format="%Y-%m-%d")).dt.days / 365.25 

    # # Add matchtype
    # data = data.merge(scomp.groupby('mastid').matchtype.min().reset_index(), how='left', on='mastid')

    # Other FEs and ID vars
    data['school_fe'] = data.groupby(['lea','schlcode']).grouper.group_info[0]
    data['school_year_fe'] = data.groupby(['lea','schlcode','year',]).grouper.group_info[0]
    data['school_year_grade_fe'] = data.groupby(['lea','schlcode','year','grade']).grouper.group_info[0]

    # Drop missing current math and reading scores
    data = data.loc[data.mathscal.notnull() & data.readscal.notnull()]

    # # teachers with at least 15 kids but no more than 200 kids in a school-year-grade
    # data = data.loc[data.groupby(['teachid','lea','schlcode','year','grade']).mastid.transform('nunique').between(15,200)]

    # Lag score missing and square
    data['miss_lag1_mathscal'] = data.lag1_mathscal.isnull().astype(int)
    data['miss_lag2_mathscal'] = data.lag2_mathscal.isnull().astype(int)
    data['miss_lag1_readscal'] = data.lag1_readscal.isnull().astype(int)
    data['miss_lag2_readscal'] = data.lag2_readscal.isnull().astype(int)
    for k in range(2,4):
        for lag in range(1,3):
            for subj in ['mathscal','readscal']:
                data['lag{}_{}_{}'.format(lag,subj,k)] = data['lag{}_{}'.format(lag,subj)]**k

    # SGY ans S means of controls
    for control in ['black','white','female','disadv','lim_eng','pared_nohs','pared_hsorless','pared_somecol','pared_baormore',
                'exc_not','exc_aig','exc_behav','exc_educ','grade_rep',
                'lag1_mathscal','lag1_readscal','lag1_mathscal_2','lag1_readscal_2','lag1_mathscal_3','lag1_readscal_3',
                'miss_lag1_mathscal','miss_lag1_readscal',
                'lag2_mathscal','lag2_readscal','lag2_mathscal_2','lag2_readscal_2','lag2_mathscal_3','lag2_readscal_3',
                'miss_lag2_mathscal','miss_lag2_readscal']:
        print("Working on means for {}".format(control))
        data['sgy_mean_{}'.format(control)] = data.groupby('school_year_grade_fe')[control].transform('mean')
        data['s_mean_{}'.format(control)] = data.groupby('school_fe')[control].transform('mean')
            # NB these are missing of non-missing values for each control
            # will need to fill in means where all obs are missing control in stata using whatever method is preferred

    # Add year lags of sgy_means
    tolag = [c for c in data.columns if 'sgy_mean' in c]
    tmp = data[['mastid','year'] + tolag].rename(columns=lambda x: 'lag_' + x if 'sgy_mean' in x else x)
    tmp['year'] = tmp.year + 1
    data = data.merge(tmp, how='left', on=['mastid','year'])

    # Save
    data.to_stata('/scratch/public/ncrime/tmp.dta', write_index=False) 

### 12) Build final analysis dataset
if 1:
    # Load the data
    data = pd.read_stata('/scratch/public/ncrime/tmp.dta')

    ## Drop observations with no teacher of any type
    data = data.loc[data.teachid_hr.notnull() | data.teachid_math.notnull() | data.teachid_eng.notnull()]

    ## Convert to long format and relabel
    data['n_teach'] = data.teachid_hr.notnull().astype(int) + data.teachid_math.notnull().astype(int) + data.teachid_eng.notnull().astype(int)
    data = data.loc[data.index.repeat(data.n_teach)].reset_index(drop=True)
    data['teach_order'] = 1
    data['teach_order'] = data.groupby(['mastid','year']).teach_order.cumsum()

    data.loc[(data.teach_order == 1) & data.teachid_hr.notnull(), 'subject'] = 'hr'
    data.loc[(data.teach_order == 1) & data.teachid_hr.notnull(), 'teachid'] = data.teachid_hr

    data.loc[(data.teach_order == 1) & data.subject.isnull() & data.teachid_math.notnull(), 'subject'] = 'math'
    data.loc[(data.teach_order == 1) & data.teachid.isnull() & data.teachid_math.notnull(), 'teachid'] = data.teachid_math

    data.loc[(data.teach_order == 1) & data.subject.isnull() & data.teachid_eng.notnull(), 'subject'] = 'eng'
    data.loc[(data.teach_order == 1) & data.teachid.isnull() & data.teachid_eng.notnull(), 'teachid'] = data.teachid_eng

    data.loc[(data.teach_order == 2) & data.subject.isnull() & data.teachid_math.notnull() & data.teachid_hr.notnull(), 'subject'] = 'math'
    data.loc[(data.teach_order == 2) & data.teachid.isnull() & data.teachid_math.notnull() & data.teachid_hr.notnull(), 'teachid'] = data.teachid_math

    data.loc[(data.teach_order == 2) & data.subject.isnull() & data.teachid_eng.notnull(), 'subject'] = 'eng'
    data.loc[(data.teach_order == 2) & data.teachid.isnull() & data.teachid_eng.notnull(), 'teachid'] = data.teachid_eng

    data.loc[(data.teach_order == 3) & data.subject.isnull() & data.teachid_eng.notnull(), 'subject'] = 'eng'
    data.loc[(data.teach_order == 3) & data.teachid.isnull() & data.teachid_eng.notnull(), 'teachid'] = data.teachid_eng

    # Student-year-subject level data
    assert data.teachid.isnull().sum() == 0
    assert data.duplicated(['mastid','year','subject']).max() == 0
    print(data.loc[data.mastid == np.random.choice(data.mastid.unique()), ['mastid','year','teachid','subject','teachid_hr','teachid_eng','teachid_math']])
    data['teachid_year_subject_fe'] = data.groupby(['teachid','year','subject']).grouper.group_info[0]
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N for valid teachers {}".format(orig_shape))

    ## Sample selections
    # Teachers observed in multiple schools in single year
    data = data.loc[data.groupby(['teachid','year']).school_fe.transform('nunique') == 1]
    new_shape =data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after single schools {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]

    # Teachers observed in single grade
    data = data.loc[data.groupby(['teachid','year']).grade.transform('nunique') == 1]
    new_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after single grade {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    
    # Drop special schools
    for col in ['alt_schl','spec_ed_schl']:
        data = data.loc[data[col] == 0]
    new_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after no special schools {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]

    # Keep obs with non-missing lag scores or in grade 3
    data = data.loc[(data.lag1_mathscal.notnull() & data.lag1_readscal.notnull())
        | (data.grade == 3)]
    new_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after has lag {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]

    # Restrict to 15 to 100 students per teacher-year-grade-subejct
    data = data.loc[data.groupby(['teachid','year','grade','subject']).mastid.transform(
                'nunique').between(15,100)]
    new_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]
    print("N after 15-100 students {}".format(new_shape))
    orig_shape = data.loc[(data.grade >= 4) & data.year.between(1997,2013)].shape[0]

    # Add pvalues for sorting
    if 0: 
        from scipy.stats import chi2_contingency

        # Add pvalues for grades 4-8 and 2007 on
        def chi2(table):
            for col in table.columns:
                if table[col].sum() == 0:
                    table = table.drop(col, axis=1)
            table = table.values
            r, p, dof, e = chi2_contingency(table)
            return p

        def classroom_chi2(c):
            current = data.loc[data.school_year_grade_fe == c]
            lea = data.loc[data.school_year_grade_fe == c].lea.values[0]
            schl = data.loc[data.school_year_grade_fe == c].schlcode.values[0]
            yr = data.loc[data.school_year_grade_fe == c].year.values[0]
            grd = data.loc[data.school_year_grade_fe == c].grade.values[0]
            lag = data.loc[(data.year == current.year.max() - 1) &
                                data.mastid.isin(current.mastid), ['mastid','teachid_year_subject_fe']].rename(
                        columns={'teachid_year_subject_fe':'lag_teachid_year_subject_fe'})
            current = current[['mastid','teachid_year_subject_fe']].merge(lag, how='inner', on='mastid')
            table = pd.crosstab(current.teachid_year_subject_fe, current.lag_teachid_year_subject_fe)
            try:
                p = chi2(table)
            except:
                p = None
            print(c,table,p,"\n", flush=True)
            return lea, schl, yr, grd, p

        gc.collect()
        import multiprocessing
        if __name__ == "__main__":
            p = multiprocessing.Pool(7)
            ans = p.map(classroom_chi2, data.school_year_grade_fe.unique())

        pvals = pd.DataFrame(ans, columns=['lea','schlcode','year','grade','pval'])
        pvals.to_pickle('/scratch/public/ncrime/pvals_post.pkl')

        # Make histogram
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="white", color_codes=True)
        fig, axes = plt.subplots()
        sns.histplot(data=pvals, x="pval", stat='probability', ax=axes)
        axes.set_xlabel('P-value') 
        axes.set_ylabel('Density of school-grade-year observations') 
        fig.tight_layout()
        fig.savefig('/accounts/projects/crwalters/cncrime/teachers_final/figures/pvalue_histogram.pdf')
        fig, axes = plt.subplots()
        sns.histplot(data=pvals.loc[pvals.pval < 1], x="pval", stat='probability', ax=axes)
        axes.set_xlabel('P-value') 
        axes.set_ylabel('Density of school-grade-year observations') 
        fig.tight_layout()
        fig.savefig('/accounts/projects/crwalters/cncrime/teachers_final/figures/pvalue_histogram_l1.pdf')
    else:
        pvals = pd.read_pickle('/scratch/public/ncrime/pvals_post.pkl')

    data = data.merge(pvals, how='left', on=['lea','schlcode','year','grade'])

    # "Classroom" means of covariates
    for control in ['black','white','female','disadv','lim_eng','pared_nohs','pared_hsorless','pared_somecol','pared_baormore',
                'exc_not','exc_aig','exc_behav','exc_educ','grade_rep',
                'lag1_mathscal','lag1_readscal','lag1_mathscal_2','lag1_readscal_2','lag1_mathscal_3','lag1_readscal_3',
                'miss_lag1_mathscal','miss_lag1_readscal',
                'lag2_mathscal','lag2_readscal','lag2_mathscal_2','lag2_readscal_2','lag2_mathscal_3','lag2_readscal_3',
                'miss_lag2_mathscal','miss_lag2_readscal']:

        print("Working on control {}".format(control))
        data['sgyts_mean_{}'.format(control)] = data.groupby(['school_year_grade_fe','teachid','subject'])[control].transform('mean')

    # Teacher exit / entrance indicators
    data['teach_first_year'] = data.groupby(['teachid','school_fe','grade','subject']).year.transform('min')
    data['teacher_enter'] = (data.year == data.teach_first_year).astype(int)
    data['teach_last_year'] = data.groupby(['teachid','school_fe','grade','subject']).year.transform('max')
    data['teacher_exit'] = (data.year == data.teach_last_year).astype(int)
    data['teach_nyears'] = data.groupby(['teachid','school_fe','grade','subject']).year.transform('nunique')
    data.loc[data.year == data.year.min(), 'teacher_enter'] = 0
    data.loc[data.year == data.year.max(), 'teacher_exit'] = 0

    # Number of prior years in school-grade-subject 
    for name, grp in [('_schgrdsubj',['school_fe','grade','subject']),
                    ('_schgrd',['school_fe','grade']),
                    ('_school',['school_fe']),
                    ('',[]),]:
        # For teacher
        tmp = data[['teachid'] + grp + ['year']].drop_duplicates().sort_values(['teachid'] + grp + ['year'])
        tmp['nyears_teach{}'.format(name)] = 1
        tmp['nyears_teach{}'.format(name)] = tmp.groupby(['teachid'] + grp)['nyears_teach{}'.format(name)].cumsum() - 1
        data = data.merge(tmp, how='left', on=['teachid'] + grp +['year'])

        # For unit
        if len(grp) > 0:
            tmp = data[grp + ['year']].drop_duplicates().sort_values(grp + ['year'])
            tmp['nyears{}'.format(name)] = 1
            tmp['nyears{}'.format(name)] = tmp.groupby(grp)['nyears{}'.format(name)].cumsum() - 1
            data = data.merge(tmp, how='left', on=grp +['year'])

    # data.loc[data.teachid == np.random.choice(data.teachid.unique()),
    #     ['teachid','year','school_fe','grade','subject','teach_first_year','teach_last_year','teach_nyears']].drop_duplicates()

    # Save
    data.to_stata('/scratch/public/ncrime/analysis_11_01_2021.dta', write_index=False, convert_dates = {'bdate':'td'}) 


# Report when we can observe stuff
data = pd.read_stata('/scratch/public/ncrime/analysis_08_03_2021.dta')
print(data.groupby(['year','grade']).mathscal.apply(lambda x: x.notnull().mean()).unstack())
print(data.groupby(['year','grade']).aoc_any.apply(lambda x: x.notnull().mean()).unstack())
print(data.groupby(['year','grade']).any_discp.apply(lambda x: x.notnull().mean()).unstack())
print(data.groupby(['year','grade']).daysabs.apply(lambda x: x.notnull().mean()).unstack())
print(data.groupby(['year','grade']).mastid.nunique().unstack())

# Teacher concordencse
data['tmp'] = data.cmb_teachid_math == data.teachid_eogm
print(data.loc[data.cmb_teachid_math.notnull() & data.teachid_eogm.notnull() & (data.subject == "math")].groupby(['year','grade']).tmp.mean().unstack())

# ### 10) Put everything together
# # Load built datasets
# data = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/cmb.pkl.gz", compression='gzip').rename(
#         columns={'sex':'sex_cmb','ethnic':'ethnic_cmb'})
# demos = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/demos.pkl.gz", compression='gzip')
# eog = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/eog.pkl.gz", compression='gzip')
# eoc = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/eoc.pkl.gz", compression='gzip')
# teach_educ = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/teach_educ.pkl.gz", compression='gzip')
# teach_lic = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/teach_lic.pkl.gz", compression='gzip')
# # pvals = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/pvals.pkl.gz", compression='gzip')

# # Merge together
# ori_shape = len(data)
# data = data.merge(demos, how='left', on='mastid')
# data = data.merge(eog, how='left', on=['mastid','year','grade'])
# data = data.merge(eoc, how='left', on=['mastid','year'])

# teach_educ['educ_lvl_cd'] = pd.to_numeric(teach_educ.educ_lvl_cd)
# data = data.merge(teach_educ.groupby('teachid').educ_lvl_cd.max().reset_index(), how='left', on='teachid')
# # data = data.merge(pvals, how='left', on=['lea','schlcode','year','grade','coursetitle'])
# assert data.shape[0] == ori_shape

# # Drop alternative schools
# data = data.loc[data.voc_schl + data.spec_ed_schl + data.alt_schl == 0]



# # Fill in sex / ethnic
# data['sex'] = data.sex.where(data.sex.notnull(), data.sex_cmb)
# data['ethnic'] = data.ethnic.where(data.ethnic.notnull(), data.ethnic_cmb)
# data.drop(['sex_cmb','ethnic_cmb'], axis=1, inplace=True)

# data['female'] = (data['sex'] == 'F').astype(int)
# data['female'].where(data['sex'].notnull(), np.nan, inplace=True)

# data['black'] = (data['ethnic'] == 'B').astype(int)
# data['white'] = (data['ethnic'] == 'W').astype(int)
# data['hispanic'] = (data['ethnic'] == 'H').astype(int)
# for ll in ['black', 'white', 'hispanic']:
#     data[ll].where(data['ethnic'].notnull(), np.nan, inplace=True)


# # # Save
# # data.sort_values(['mastid','year','grade','semester','coursetitle','section','teachid'], inplace=True)
# # data.to_pickle("/accounts/projects/crwalters/cncrime/users/ekrose/data/cmb_final.pkl.gz", compression='gzip')


# ### 11) Construct analysis version
# # Relevant years / grades
# data = data.loc[data.grade.between(4,8) & (data.year >= 2007)]

# # De dup as required
# print(data.shape)
# data = data.loc[data.groupby('mastid').birthdt.transform('nunique').astype(int) == 1]  # 97.4% of mastids have a unique birthdt
# # data = data.loc[data.groupby('classroom_id').teachid.transform('nunique') == 1]     # not doing this for now, 95% of classrooms
# print(data.shape)
# data = data.loc[~data.duplicated(['classroom_id','teachid','mastid'], keep=False)]  # Drops about 20k rows
# print(data.shape)

# # Classrooms with at least 15 kids
# data = data.loc[data.groupby('classroom_id').mastid.transform('nunique') >= 15]

# # Drop missing current math and reading scores
# data = data.loc[data.mathscal.notnull() & data.readscal.notnull()]

# # Make sure there are at least 2 teachers in the sgyc
# data = data.loc[data.groupby('school_course_grade_year_fe').teachid.transform('nunique') >= 2]

# # Lag score missing and square
# data['miss_lag1_mathscal'] = data.lag1_mathscal.isnull().astype(int)
# data['lag1_mathscal_2'] = data.lag1_mathscal**2
# data['miss_lag1_readscal'] = data.lag1_readscal.isnull().astype(int)
# data['lag1_readscal_2'] = data.lag1_readscal**2

# # SGYC means of controls
# for control in ['black','white','female','disadv','pared_hsorless','pared_somecol','pared_baormore',
#             'exc_not','exc_aig','exc_behav','exc_educ',
#             'lag1_mathscal','lag1_readscal','miss_lag1_mathscal','miss_lag1_readscal']:
#     data['sgyc_mean_{}'.format(control)] = data.groupby('school_course_grade_year_fe')[control].transform('mean')
#     data['sy_mean_{}'.format(control)] = data.groupby('school_year_fe')[control].transform('mean')

# for control in ['lag1_mathscal','lag1_readscal']:
#     for power in range(2,4):
#         data['sgyc_mean_{}_{}'.format(control,power)] = data['sgyc_mean_{}'.format(control)]**power
#         data['sy_mean_{}_{}'.format(control,power)] = data['sy_mean_{}'.format(control)]**power

# # Save
# data.drop('unique_courses', axis=1).to_stata('/scratch/public/ncrime/tmp.dta', write_index=False) 



# # '''
# # * Regressions
# # eststo clear
# # eststo: reg math_index mathscal_lom sgyc_mean_* lag1_mathscal* lag1_readscal* i.year##i.grade if grade >= 4, cluster(mastid)
# # eststo: reg mathscal mathscal_lom sgyc_mean_* lag1_mathscal* lag1_readscal* i.year##i.grade if grade >= 4, cluster(mastid)
















# data['nteachers_my'] = data.groupby(['mastid','year']).teachid.transform('nunique')

# # Math
# data['math_teacher'] = data.coursecode > 2000
# data.loc[data.math_teacher, 'nmath_teachers_my'] = data.teachid
# data['nmath_teachers_my'] = data.groupby(['mastid','year']).nmath_teachers_my.transform('nunique')

# # Math
# data['math_teacher'] = data.coursecode > 2000
# data.loc[data.math_teacher, 'nmath_teachers_my'] = data.teachid
# data['nmath_teachers_my'] = data.groupby(['mastid','year']).nmath_teachers_my.transform('nunique')


# data.loc[data.mastid == np.random.choice(data.loc[(data.nmath_teachers_my > 1) & (data.grade < 7), 'mastid'])]

# data.loc[data.mastid == np.random.choice(data.mastid.unique()), ['mastid','year','grade','coursetitle','coursecode','teachid','semester','section','lea','schlcode','nmath_teachers_my']]
# data.loc[data.mastid == 7643236.0],  ['mastid','year','grade','coursetitle','coursecode','teachid','semester','section','lea','schlcode','nmath_teachers_my']]

# data['dup_class'] = data.duplicated(['lea','schlcode','year','teachid','semester','section','mastid','coursecode'], keep=False)

# data.loc[data.mastid == np.random.choice(data.loc[data.dup_class, 'mastid']), ['mastid','year','grade','coursetitle','coursecode','teachid','semester','section','lea','schlcode','dup_class']]


# # Study skills?
# # Grade repeaters
# # ld -- learning disability....
# # What if a student has multiple math or reading classes in a year?

# # #####################################
# # ### Build 4-8 grade data
# # #####################################

# # d = data.loc[data.grade.between(3,8, inclusive = True)].copy() #Grades 4-8

# # # Remove columns for EOC files
# # _tmp = list(eoc.columns)
# # _tmp.remove('mastid')
# # _tmp.remove('year')
# # d.drop(_tmp, axis=1, inplace = True)

# # del data, eoc
# # gc.collect()

# # #####################################
# # # 1) Sample restrictions:
# # #####################################



# # # Make sure that the unit of observation is: ['mastid','year','coursetitle'] by removing students who appear in multiple school within a year
# # # d = d.loc[d.groupby(['mastid', 'year'])['schlid'].transform('nunique') == 1].copy()

# # # Drop if teacher observed in multiple schools in the same year
# # d = d.loc[d.groupby(['teachid', 'year'])['schlid'].transform('nunique') == 1].copy()

# # # Drop classrooms/courses with 90% male or females (drops less than 700 students)
# # _tmp = d.groupby(['school_course_grade_year_fe','teachid'])['female'].transform('mean')
# # d = d.loc[(_tmp>0.1) & (_tmp < 0.9)]

# # # TO ADD (?): Add special student/courses information to identify tracking


# # # Remove student that re-take a grade
# # # Indicator for retaking a test
# # d['retake_grade_flag'] = (d['grade'] == d['lag_grade']).astype(int)
# # d['retake_grade_flag'].where(d['lag_grade'].notnull() & (d['lag_grade'] <= d['grade']), np.nan, inplace=True)
# # d = d.loc[d['retake_grade_flag'] != 1]
# # del d['lag_grade'], d['retake_grade_flag'] # YST: I don't think we need to drag on this variable

# # # Restrict to course-teacher pairs with at least 15 students and no more than 90 students (note a teacher can potential teach multiple classes in a year)
# # _tmp = d.groupby(['school_course_grade_year_fe','teachid'])['mastid'].transform('nunique')
# # d = d.loc[_tmp.between(15,90, inclusive = True)]


# # ##############
# # # Pre-testing
# # ##############
# # from scipy.stats import chi2_contingency

# # # At least 15 kids in class (YST: being done above, left it here for clarity, to be deleted soon)
# # # d = d.loc[d.groupby(['school_course_grade_year_fe','teachid']).mastid.transform('nunique') >= 15]

# # # At least 3 teachers  in sgyc
# # d = d.loc[d.groupby(['school_course_grade_year_fe']).teachid.transform('nunique') >= 3]

# # # Overlap statistics for each class....
# # d['classroom_id'] = d.groupby(['lea','schlcode','year','grade','coursetitle','teachid']).grouper.group_info[0]


# # def chi2(table):
# #     for col in table.columns:
# #         if table[col].sum() == 0:
# #             table = table.drop(col, axis=1)
# #     table = table.values
# #     r, p, dof, e = chi2_contingency(table)
# #     return p

# # def classroom_chi2(c):
# #     current = d.loc[d.school_course_grade_year_fe == c]
# #     lag = d.loc[(d.year == current.year.max() - 1) & # (d.schlid == current.schlid.max()) &
# #                         d.mastid.isin(current.mastid), ['mastid','classroom_id']].rename(
# #                 columns={'classroom_id':'lag_classrom_id'})
# #     current = current[['mastid','classroom_id']].merge(lag, how='inner', on='mastid')
# #     table = pd.crosstab(current.classroom_id, current.lag_classrom_id)
# #     try:
# #         p = chi2(table)
# #     except:
# #         p = None
# #     print(table,p,"\n", flush=True)
# #     return c, p

# # import multiprocessing
# # if __name__ == "__main__":
# #     p = multiprocessing.Pool(5)
# #     ans = p.map(classroom_chi2, d.loc[(d.year >= 2008) & (d.grade >= 4)].school_course_grade_year_fe.unique())

# # pvals = pd.DataFrame(ans, columns=['school_course_grade_year_fe','pval'])
# # d = d.merge(pvals, how='left', on=['school_course_grade_year_fe'])
# # d0 = d.copy()
# # d = d.loc[d.pval > 0.05]

# # ####################################################################################
# # # # Re-iterate to make sure that the following 2 restrictions are satisfied:
# # # 1. Each teacher appears in more than one year
# # # 2. Each School-grade-year combo has more than 3 teachers
# # ####################################################################################

# # _tmp_nteach_years = 0
# # _tmp_nteach_cluster = 0
# # itr = 0
# # while (_tmp_nteach_years==0) | (_tmp_nteach_cluster==0):
# #     print('Working on iteration {}, sample size: {} \n'.format(itr, int(d.shape[0])), flush=True)

# #     # Teachers appear in X years (i.e., at least 2 years)
# #     _nteach_years = d.groupby(['teachid'])['year'].transform('nunique')
# #     _tmp_nteach_years = _nteach_years.min() > 1
# #     d = d.loc[_nteach_years >= 2]

# #     # Restrict to school-grade-year combos with at least 3 valid teachers
# #     _nteach_cluster = d.groupby(['school_course_grade_year_fe'])['teachid'].transform('nunique')
# #     _tmp_nteach_cluster = _nteach_cluster.min() >= 3
# #     d = d.loc[_nteach_cluster >= 3]
# #     itr += 1






# # ### Add mid-term outcomes
# # # 1) Grade took each of the following classes
# # for col in ['alg1','alg2','geom']:

# #     # Score when first took course and year in which first took course
# #     _tmp = eoc.loc[eoc[col+'scal'].notnull(), ['mastid', 'year', col+'scal']].copy()
# #     _tmp.sort_values(['mastid', 'year'], ascending = [True, True], inplace = True)
# #     _tmp.drop_duplicates(subset = ['mastid'], keep = 'first', inplace=True)
# #     _tmp.rename(columns = {'year':'yr_took_'+col}, inplace=True)
# #     d = d.merge(_tmp, how = 'left', on = ['mastid'])
# #     del _tmp


# # # 2) Honor / AP classes? 


# # # 3) 


# # # Save
# # d.to_stata("/accounts/projects/crwalters/cncrime/users/ekrose/data/cmb_gr4to8_analysis_pretest.dta")




# # ### Some checks
# # # 1) Make sure that a student meets a teacher only once
# # # _tmp = d[['mastid','year','teachid']].drop_duplicates()
# # # a = _tmp.groupby(['mastid'])['teachid'].count()
# # # aa = _tmp.groupby(['mastid'])['teachid'].nunique()
# # # In [68]: (a == aa).value_counts(normalize=True)                                                                      
# # # Out[68]: 
# # # True     0.979724
# # # False    0.020276


















# # ### Put things together
# #     # Add EOG to main build
# #     assert eog.duplicated(['mastid','year','grade']).sum() == 0
# #     data = data.merge(eog.rename(columns={
# #                 'grade':'eog_grade'}), how='left', on=['mastid','year','grade']) 



# # ### 4) Add EOC data
# # if 1:
# #     print("\n\nAdding EOC files")
# #     data = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/wmorrison/data/cmb.pkl.gz", compression='gzip')
# #     eogs = []
# #     for year in range(firstyr-2,lastyr+1):
# #         for grade in range(3,9):
# #             print("Loading EOG for {} grade {}".format(year,grade))
# #             with gzip.open(data_path + "/Student/End of Grade/{}/eog{}pub{}.dta.gz".format(year,grade,str(year)[-2:]), 'rb') as inF:
# #                 eog = pd.read_stata(inF)


# # # Save it
# # data.loc[data.cansee_sc1621 == 1].to_stata('/scratch/public/ncrime/tmp.dta', write_index=False) 


# # '''
# # 22k teachers, 
# # '''

# # '''
# # use "/scratch/public/ncrime/tmp.dta", clear

# # * LOM crime
# # bys teachid: egen _tot = total(anysc_1621)
# # bys teachid year: egen _totyear = total(anysc_1621)
# # bys teachid: egen _count = count(anysc_1621)
# # bys teachid year: egen _countyear = count(anysc_1621)
# # gen lom = (_tot - _totyear) / (_count - _countyear)
# # egen teachid_tag = tag(teachid)
# # su _count if teachid_tag, d
# # drop _*

# # * Covariates
# # gen male = sex == "M"

# # * Simple regressions
# # eststo clear
# # eststo: reg anysc_1621 lom, cluster(teachid)
# # eststo: areg anysc_1621 lom, absorb(school_course_fe) cluster(teachid)
# # eststo: areg anysc_1621 lom, absorb(school_course_grade_fe) cluster(teachid)
# # eststo: areg anysc_1621 lom, absorb(school_course_grade_year_fe) cluster(teachid)

# # eststo: reg male lom, cluster(teachid)
# # eststo: areg male lom, absorb(school_course_fe) cluster(teachid)
# # eststo: areg male lom, absorb(school_course_grade_fe) cluster(teachid)
# # eststo: areg male lom, absorb(school_course_grade_year_fe) cluster(teachid)



# # '''

# # ### 1) Do build for english and algebra teachers
# # if 1:
# #     finals = []
# #     for year in range(firstyr,lastyr+1):
# #         for subj in [["1021"],["2021","2022","2023"],["2024"]]:
# #             print("Working on subject {}".format(subj))

# #             ## 1) Load up the SAR files
# #             print("Loading SAR for {}".format(year))
# #             with gzip.open(data_path + "/Classroom/Personnel/pers{}.dta.gz".format(year), 'rb') as inF:
# #                 teachers = pd.read_stata(inF)

# #             # Keep english 1 teachers 
# #             teachers = teachers.loc[teachers['astype'] == "TE"]
# #             teachers = teachers.loc[teachers.subjct.isin(subj)]

# #             # Subset to spring semester assignment
# #             teachers = teachers.loc[teachers.semstr == "2"]


# #             ## 2) Add student counts
# #             print("Loading student counts for {}".format(year))
# #             with gzip.open(data_path + "/Classroom/Student Count/studir{}.dta.gz".format(year), 'rb') as inF:
# #                 counts = pd.read_stata(inF)

# #             # Subset and sum counts data
# #             counts = counts.loc[pd.to_numeric(counts.semstr) == 2]
# #             merge_vars = ['lea','schlcode','crsnum','secnum']
# #             counts = counts.loc[counts.subjct.isin(subj)]
# #             counts_orig = counts.copy()
# #             counts['tot09'] = (counts.grdlvl == "09")*counts.totcnt
# #             counts = counts.groupby(merge_vars)[['totcnt','tot09',
# #                     'whtef','whtem','blckf','blckm','hispf','hispm','multf','multm','aminf','aminm','asiaf','asiam',
# #                     'acadgft','nonexc','bemh']].sum().reset_index()
# #             assert counts.duplicated(merge_vars).sum() == 0

# #             # Merge onto teachers
# #             teachers = teachers.merge(counts, how='left', on=merge_vars)
# #             teachers = teachers.merge(counts_orig.groupby(merge_vars).crstitle.first().reset_index(), 
# #                         how='left', on=merge_vars)
            

# #             ## 3) Subset teachers as desired
# #             # Keep assignments with at least 15 students
# #             teachers = teachers.loc[teachers.totcnt >= 15]

# #             # Construct chi-squre contingency tests
# #             from scipy.stats import chi2_contingency
# #             teachers['cell'] = teachers.groupby(['lea','schlcode','crstitle']).grouper.group_info[0]

# #             def chi2(x):
# #                 table = x[['whtef','whtem','blckf','blckm','hispf','hispm','multf','multm','aminf','aminm','asiaf','asiam']]
# #                 for col in table.columns:
# #                     if table[col].sum() == 0:
# #                         table = table.drop(col, axis=1)
# #                 table = table.values
# #                 r, p, dof, e = chi2_contingency(table)
# #                 return p

# #             chi2ps = teachers.groupby('cell').apply(chi2)
# #             chi2ps.name = "chi2p"
# #             teachers = teachers.merge(chi2ps.reset_index(), how='left', on='cell')

# #             # Variables we'll add here 
# #             toadd = ['totcnt','tot09','whtef','whtem','blckf','blckm','acadgft','nonexc','bemh','chi2p']

# #             ## 4) Load up course membership files
# #             print("Loading course membership for {}".format(year))
# #             with gzip.open(data_path + "/Student/Course Membership/crs_memb_pub{}.dta.gz".format(year,str(year)[-2:]), 'rb') as inF:
# #                 cmb = pd.read_stata(inF)
# #             cmb['id'] = cmb.reset_index().index

# #             # Keep just spring semester
# #             cmb = cmb.loc[cmb.semester == "2"]

# #             # Merge on classrooms strictly
# #             teachers = teachers.reset_index()
# #             teachers['section'] = pd.to_numeric(teachers.secnum)
# #             cmb['section'] = pd.to_numeric(cmb.section, errors='coerce')
# #             teachers['localcourse'] = teachers.crsnum
# #             assert teachers.duplicated(['lea','schlcode','localcourse','section']).sum() == 0
# #             matches = cmb.drop('teachid', axis=1).merge(teachers[['index','lea','schlcode','localcourse','section','teachid','acadlvl'] + toadd],
# #                         how='inner', on=['lea','schlcode','localcourse','section'])
# #             print("Share of clasrooms matched exactly: {:4.3f}".format(teachers['index'].isin(matches['index']).mean()))
# #             cmb = cmb.loc[~cmb.id.isin(matches.id)]

# #             # Now merge using destring course
# #             teachers['course'] = pd.to_numeric(teachers.crsnum.str.replace('[^0-9]',''))
# #             missing = teachers.loc[~teachers['index'].isin(matches['index'])].copy()
# #             cmb['course'] = pd.to_numeric(cmb.localcourse.str.replace('[^0-9]',''))
# #             missing = missing.loc[~missing.duplicated(['lea','schlcode','course','section'], keep=False)]

# #             matches = pd.concat((matches,
# #                     cmb.drop('teachid', axis=1).merge(missing[['index','lea','schlcode','course','section','teachid','acadlvl'] + toadd],
# #                         how='inner', on=['lea','schlcode','course','section'])),
# #                     sort=False)
# #             print("Share of clasrooms matched using destring courses: {:4.3f}".format(teachers['index'].isin(matches['index']).mean()))
# #             cmb = cmb.loc[~cmb.id.isin(matches.id)]

# #             # Still missing! Just use section and teacher
# #             missing = teachers.loc[~teachers.index.isin(matches['index'])].copy()
# #             missing = missing.loc[~missing.duplicated(['lea','schlcode','teachid','section'], keep=False)]
# #             matches = pd.concat((matches,
# #                     cmb.merge(missing[['index','lea','schlcode','course','section','teachid','acadlvl'] + toadd],
# #                         how='inner', on=['lea','schlcode','teachid','section'])),
# #                     sort=False)
# #             print("Share of clasrooms matched using section / teacher: {:4.3f}".format(teachers['index'].isin(matches['index']).mean()))
# #             cmb = cmb.loc[~cmb.id.isin(matches.id)]

# #             # Using last digit removed
# #             missing = teachers.loc[~teachers.index.isin(matches['index'])].copy()
# #             missing['localcourse'] = missing.crsnum.apply(lambda x: x[:5])
# #             missing = missing.loc[~missing.duplicated(['lea','schlcode','localcourse','section'], keep=False)]

# #             matches = pd.concat((matches,
# #                     cmb.drop('teachid', axis=1).merge(missing[['index','lea','schlcode','localcourse','section','teachid','acadlvl'] + toadd],
# #                         how='inner', on=['lea','schlcode','localcourse','section'])),
# #                     sort=False)
# #             print("Share of clasrooms matched removing last digit: {:4.3f}".format(teachers['index'].isin(matches['index']).mean()))
# #             # cmb = cmb.loc[~cmb.id.isin(matches.id)]

# #             ## 4) Final file 
# #             final = matches.loc[:, ['mastid','sex','ethnic','grade','lea','schlcode','localcourse','section','teachid','coursetitle',
# #                                     'acadlvl'] + toadd]
# #             final['year'] = year
# #             final = final.drop_duplicates()
# #             finals += [final.copy(),]

# #     ### 5) Merge on stuff from main build
# #     # Put together CMB build
# #     data = pd.concat(finals, sort=False)
# #     data = data.loc[data.mastid.notnull()]
# #     data.to_pickle("/accounts/projects/crwalters/cncrime/users/wmorrison/data/cmb_eoc.pkl.gz", compression='gzip')

# # else:
# #     print("Loading CMB build")
# #     data = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/wmorrison/data/cmb_eoc.pkl.gz", compression='gzip')


# # ### 2) Do build for 5th grade
# # firstyr = 2007  # no 5th graders in 2006 CMB for some reason
# # lastyr = 2011
# # if 1:
# #     finals = []
# #     for year in range(firstyr,lastyr+1):
# #         for subj in [["0000","1010","2001","3001","4001"]]:
# #             print("Working on subject {}".format(subj))

# #             ## 1) Load up the SAR files
# #             print("Loading SAR for {}".format(year))
# #             with gzip.open(data_path + "/Classroom/Personnel/pers{}.dta.gz".format(year), 'rb') as inF:
# #                 teachers = pd.read_stata(inF)

# #             # Keep english 1 teachers 
# #             teachers = teachers.loc[teachers['astype'] == "TE"]
# #             teachers = teachers.loc[teachers.subjct.isin(subj)]

# #             # Subset to fall semester assignment | there almost all 1
# #             teachers['semstr'] = pd.to_numeric(teachers.semstr)
# #             teachers = teachers.loc[teachers.semstr == teachers.groupby(
# #                     ['lea','schlcode','crsnum','secnum']).semstr.transform('max')]

# #             ## 2) Add student counts
# #             print("Loading student counts for {}".format(year))
# #             with gzip.open(data_path + "/Classroom/Student Count/studir{}.dta.gz".format(year), 'rb') as inF:
# #                 counts = pd.read_stata(inF)

# #             # Subset and sum counts data
# #             counts = counts.loc[counts.subjct.isin(subj)]
# #             counts['semstr'] = pd.to_numeric(counts.semstr)

# #             merge_vars = ['lea','schlcode','crsnum','secnum','semstr']
# #             counts_orig = counts.copy()
# #             counts['tot_grade'] = (counts.grdlvl == "05")*counts.totcnt
# #             counts = counts.groupby(merge_vars)[['totcnt','tot_grade',
# #                     'whtef','whtem','blckf','blckm','hispf','hispm','multf','multm','aminf','aminm','asiaf','asiam',
# #                     'acadgft','nonexc','bemh']].sum().reset_index()
# #             assert counts.duplicated(merge_vars).sum() == 0

# #             # Merge onto teachers
# #             teachers = teachers.merge(counts, how='left', on=merge_vars)
# #             teachers = teachers.merge(counts_orig.groupby(merge_vars).crstitle.first().reset_index(), 
# #                         how='left', on=merge_vars)
            
# #             # Save fifth grade glassroms
# #             teachers = teachers.loc[teachers.tot_grade / teachers.totcnt >= 0.75]

# #             ## 3) Subset teachers as desired
# #             # Keep assignments with at least 15 students
# #             teachers = teachers.loc[teachers.totcnt >= 15]

# #             # Construct chi-squre contingency tests
# #             from scipy.stats import chi2_contingency
# #             teachers['cell'] = teachers.groupby(['lea','schlcode','crstitle','semstr']).grouper.group_info[0]

# #             def chi2(x):
# #                 table = x[['whtef','whtem','blckf','blckm','hispf','hispm','multf','multm','aminf','aminm','asiaf','asiam']]
# #                 for col in table.columns:
# #                     if table[col].sum() == 0:
# #                         table = table.drop(col, axis=1)
# #                 table = table.values
# #                 r, p, dof, e = chi2_contingency(table)
# #                 return p

# #             chi2ps = teachers.groupby('cell').apply(chi2)
# #             chi2ps.name = "chi2p"
# #             teachers = teachers.merge(chi2ps.reset_index(), how='left', on='cell')

# #             # Variables we'll add here 
# #             toadd = ['subjct','totcnt','tot_grade','whtef','whtem','blckf','blckm','acadgft','nonexc','bemh','chi2p']

# #             # Subject lengths
# #             print(teachers.subjlen.value_counts())

# #             ## 4) Load up course membership files
# #             print("Loading course membership for {}".format(year))
# #             with gzip.open(data_path + "/Student/Course Membership/crs_memb_pub{}.dta.gz".format(year,str(year)[-2:]), 'rb') as inF:
# #                 cmb = pd.read_stata(inF)
# #             cmb['id'] = cmb.reset_index().index

# #             # Keep just spring semester
# #             cmb['semstr'] = pd.to_numeric(cmb.semester, errors='coerce')

# #             # Merge on classrooms strictly
# #             teachers = teachers.reset_index()
# #             teachers['section'] = pd.to_numeric(teachers.secnum)
# #             cmb['section'] = pd.to_numeric(cmb.section, errors='coerce')
# #             teachers['localcourse'] = teachers.crsnum
# #             assert teachers.duplicated(['lea','schlcode','localcourse','section']).sum() == 0
# #             matches = cmb.drop('teachid', axis=1).merge(teachers[['index','lea','schlcode','localcourse','section','teachid','acadlvl'] + toadd],
# #                         how='inner', on=['lea','schlcode','localcourse','section'])
# #             print("Share of clasrooms matched exactly: {:4.3f}".format(teachers['index'].isin(matches['index']).mean()))
# #             cmb = cmb.loc[~cmb.id.isin(matches.id)]

# #             # Now merge using destring course
# #             teachers['course'] = pd.to_numeric(teachers.crsnum.str.replace('[^0-9]',''))
# #             missing = teachers.loc[~teachers['index'].isin(matches['index'])].copy()
# #             cmb['course'] = pd.to_numeric(cmb.localcourse.str.replace('[^0-9]',''))
# #             missing = missing.loc[~missing.duplicated(['lea','schlcode','course','section'], keep=False)]

# #             matches = pd.concat((matches,
# #                     cmb.drop('teachid', axis=1).merge(missing[['index','lea','schlcode','course','section','teachid','acadlvl'] + toadd],
# #                         how='inner', on=['lea','schlcode','course','section'])),
# #                     sort=False)
# #             print("Share of clasrooms matched using destring courses: {:4.3f}".format(teachers['index'].isin(matches['index']).mean()))
# #             cmb = cmb.loc[~cmb.id.isin(matches.id)]

# #             # Still missing! Just use section and teacher
# #             missing = teachers.loc[~teachers.index.isin(matches['index'])].copy()
# #             missing = missing.loc[~missing.duplicated(['lea','schlcode','teachid','section'], keep=False)]
# #             matches = pd.concat((matches,
# #                     cmb.merge(missing[['index','lea','schlcode','course','section','teachid','acadlvl'] + toadd],
# #                         how='inner', on=['lea','schlcode','teachid','section'])),
# #                     sort=False)
# #             print("Share of clasrooms matched using section / teacher: {:4.3f}".format(teachers['index'].isin(matches['index']).mean()))
# #             cmb = cmb.loc[~cmb.id.isin(matches.id)]

# #             # Using last digit removed
# #             missing = teachers.loc[~teachers.index.isin(matches['index'])].copy()
# #             missing['localcourse'] = missing.crsnum.apply(lambda x: x[:5])
# #             missing = missing.loc[~missing.duplicated(['lea','schlcode','localcourse','section'], keep=False)]

# #             matches = pd.concat((matches,
# #                     cmb.drop('teachid', axis=1).merge(missing[['index','lea','schlcode','localcourse','section','teachid','acadlvl'] + toadd],
# #                         how='inner', on=['lea','schlcode','localcourse','section'])),
# #                     sort=False)
# #             print("Share of clasrooms matched removing last digit: {:4.3f}".format(teachers['index'].isin(matches['index']).mean()))
# #             cmb = cmb.loc[~cmb.id.isin(matches.id)]

# #             ## 4) Final file 
# #             final = matches.loc[:, ['mastid','sex','ethnic','grade','lea','schlcode','localcourse','section','teachid','coursetitle',
# #                                     'acadlvl'] + toadd]
# #             final['year'] = year
# #             final = final.drop_duplicates()
# #             finals += [final.copy(),]

# #     ### 5) Merge on stuff from main build
# #     # Put together CMB build
# #     data = pd.concat(finals, sort=False)
# #     data = data.loc[data.mastid.notnull()]
# #     data.to_pickle("/accounts/projects/crwalters/cncrime/users/wmorrison/data/cmb_5g.pkl.gz", compression='gzip')

# #     with gzip.open("/accounts/projects/crwalters/cncrime/users/wmorrison/data/temp.dta.gz", 'rb') as inF:
# #         main = pd.read_stata(inF)
# # else:
# #     print("Loading CMB build")
# #     data = pd.read_pickle("/accounts/projects/crwalters/cncrime/users/wmorrison/data/cmb_5g.pkl.gz", compression='gzip')



# # # Generate school-by-coursetitle vars
# # data['course_schl_id'] = data.groupby(['lea','schlcode','coursetitle']).grouper.group_info[0]
# # data['course_schl_year_id'] = data.groupby(['lea','schlcode','coursetitle','year']).grouper.group_info[0]
# # data['schl_id'] = data.groupby(['lea','schlcode']).grouper.group_info[0]

# # # Save to stata
# # data.to_stata("/accounts/projects/crwalters/cncrime/users/wmorrison/data/cmb_eoc.dta")
# # '''
# # gzip "/accounts/projects/crwalters/cncrime/users/wmorrison/data/cmb_eoc.dta"
# # '''


# # '''
# # gzuse "/accounts/projects/crwalters/cncrime/users/wmorrison/data/cmb_eoc.dta.gz", clear

# # * Valid teacher
# # tostring teachid, gen(teach_string)
# # gen valid_teacher = length(teach_string) >= 6
# # keep if valid_teacher == 1
# # keep if inlist(grade,"09","10","11")

# # * Observations per teacher year
# # bys teachid course_schl_year_id: egen n_students = count(mastid)
# # keep if n_students >= 15

# # * Number of teachers per cell
# # egen _tmp = tag(teachid course_schl_year_id)
# # bys course_schl_year_id: egen tot_teachers = total(_tmp)
# # drop _*

# # * Covariates
# # gen male = sex == "M"
# # egen schl_id_tag = tag(schl_id)
# # egen course_schl_year_id_tag = tag(course_schl_year_id)
# # egen course_schl_year_teach_tag = tag(course_schl_year_id teachid)

# # * Do teachers predict 
# # levelsof course_schl_year_id, local(schls)
# # capture drop fstat*
# # gen fstat = .
# # foreach scl of local schls {
# #     di "Working on `scl'"
# #     capture {
# #     reg male i.teachid if course_schl_year_id == `scl', robust
# #     testparm i.teachid
# #     replace fstat = r(p) if course_schl_year_id == `scl'
# #     }
# # }


# # gsort - course_schl_year_teach_tag + fstat course_schl_year_id
# # list course_schl_year_id fstat year teachid coursetitle totcnt whtef whtem blckf blckm nonexc in 1/25

# # reg male i.teachid if course_schl_year_id == 1907, robust

# # binscatter fstat evar if schl_id_tag
# # hist fstat if schl_id_tag & evar < 0.02, width(0.05) frac
# # graph export hist.pdf, replace

# # areg male i.teachid#i.year if lea == "410" & schlcode == "556", absorb(course_schl_year_id) robust

# # * Merge on main stuff
# # preserve
# # gzuse "/accounts/projects/crwalters/cncrime/users/wmorrison/data/main_build.dta.gz", clear
# # keep mastid year mathscal_lag1 mathscal_lag2 readscal_lag1 readscal_lag2 teachid_eng1 test_score_eng1 testdt_eng1
# # keep if inrange(year,2006,2011)
# # tempfile tomerge
# # save `tomerge'
# # restore

# # merge m:1 mastid year using `tomerge', nogen keep(1 3)

# # * Merge on demographis
# # preserve
# # gzuse "/accounts/projects/crwalters/cncrime/users/wmorrison/data/main_build.dta.gz", clear
# # collapse (max) female black pared_college disadv aoc20any, by(mastid)
# # tempfile tomerge
# # save `tomerge'
# # restore

# # merge m:1 mastid using `tomerge', nogen keep(1 3)

# # * Teacher-level 
# # gen crime = aoc20any
# # replace crime = 0 if crime == .
# # bys teachid: egen total_crime = total(crime)
# # bys teachid: egen total_obs = count(crime)
# # bys teachid year: egen total_crime_year = total(crime)
# # bys teachid year: egen total_obs_year = count(crime)
# # gen crime_loom = (total_crime - total_crime_year) / (total_obs - total_obs_year)

# # * generate crime index
# # foreach var of varlist black male pared_college disadv {
# #     qui su `var'
# #     replace `var' = r(mean) if missing(`var') 
# # }
# # reg crime black male pared_college disadv
# # predict crime_index, xb

# # * Output regressions
# # eststo clear
# # eststo: reg crime_index crime_loom, robust
# # eststo: areg crime_index crime_loom, absorb(course_schl_year_id) robust
# # estadd local fes = "\checkmark"
# # eststo: areg male crime_loom, absorb(course_schl_year_id) robust
# # estadd local fes = "\checkmark"
# # estadd local tossout = "\checkmark"
# # eststo: areg crime_index crime_loom if chi2p >= 0.1 & grade == "09", absorb(course_schl_year_id) robust
# # estadd local fes = "\checkmark"
# # estadd local tossout = "\checkmark"
# # estadd local ninth = "\checkmark"
# # eststo: areg crime crime_loom if grade == "09" & chi2p >= 0.1, absorb(course_schl_year_id) robust
# # estadd local fes = "\checkmark"
# # estadd local tossout = "\checkmark"
# # estadd local ninth = "\checkmark"

# # esttab, keep(crime_loom) se stats(fes tossout ninth N, labels("Course-school-year FE" "Remove Chi2 pvals < 0.2" "Ninth graders only" "Observations")) 


# # '''



