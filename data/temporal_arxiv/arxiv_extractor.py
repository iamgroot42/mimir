from utils import *
import magic
mime = magic.Magic(mime=True)
import re
import multiprocessing as mp
import chardet
import bs4
import time


sh("mkdir -p tmp tmp2 out done fallback_needed errored")


def any_to_utf8(b):
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError: 
        # try to figure out encoding if not urf-8

        guess = chardet.detect(b)['encoding']

        if not guess or guess == 'UTF-8': return

        try:
            return b.decode(guess)
        except (UnicodeDecodeError, LookupError):
            # still cant figure out encoding, give up
            return

def convert(tex):
    print(tex)
    out_name = tex.split('/')[2:] >> join('_')

    try:
        with open(tex, 'rb') as fh:
            b = fh.read()
            cont = any_to_utf8(b)
            if cont is None: return
        fwrite(tex, cont)
    except FileNotFoundError:
        # ???
        
        return

    try:
        sh(f'timeout 10s pandoc -s {tex} -o out/{out_name}.md  --wrap=none')
        print(os.path.exists(f'out/{out_name}.md'))
    except ExitCodeError:
        import traceback
        traceback.print_exc()
        # fallback:
        try:
            # move to fallback pile so we can handle it later
            if '_extract' in tex.split('/')[:-1] >> join('/'):
                loc = tex.split('/')[:-1] >> join('/')
            else:
                loc = tex
            sh(f'mv {loc} fallback_needed/')

            return

        except ExitCodeError:
            import traceback
            traceback.print_exc()


def preextract_tar(dump):
    dump_name = dump.split('/')[-1][:-4]
    sh(f"(mkdir -p tmp2/{dump_name}; tar xf {dump} -C tmp2/{dump_name} && touch tmp2/done_{dump_name}; echo finished preload of {dump_name}) &")

def copy_tar(dump):
    dump_name = dump.split('/')[-1][:-4]
    for i in range(120):
        if os.path.exists(f'tmp2/done_{dump_name}'):
            sh(f'mv tmp2/{dump_name}/* tmp')
            return True
        print('waiting for tar...')
        time.sleep(1)

    return False

pool = mp.Pool(12)

files = ls('files')

sh("rm -rf tmp/* tmp2/*")
preextract_tar(files[0])

for i, dump in enumerate(tqdm(files)):
    if i + 1 < len(files): preextract_tar(files[i + 1])
    try:
        sh("rm -rf tmp/*")
        if not copy_tar(dump): continue
        # extract
        print(dump)
        sh(f"tar xf {dump} -C tmp")

        for doc in lsr('tmp'):
            if doc.endswith('.gz'):
                sh(f"gunzip {doc}")
                type = mime.from_file(doc[:-3])
    #            print(type)
                if type == 'application/x-tar':
                    sh(f"mkdir -p {doc[:-3]}_extract && tar xf {doc[:-3]} -C {doc[:-3]}_extract")
                    sh(f"rm {doc[:-3]}")
                elif type == 'text/x-tex':
                    sh(f"mv {doc[:-3]} {doc[:-3]}.tex")
                else:
                    sh(f"rm {doc[:-3]}")

            elif doc.endswith('.pdf'):
                sh(f"rm {doc}")

        # process

        def tex_files():
            for doc in ls(ls('tmp')[0]):
                if os.path.isdir(doc):
                    for name in ['main', 'Main', 'MAIN', 'paper', 'Paper']: # common main file names
                        if os.path.exists(doc + '/' + name + '.tex'):
                            yield doc + '/' + name + '.tex'
                            break
                    else:
                        if ls(doc) >> filt(X.endswith('.tex')) >> apply(len) == 1:
                            yield ls(doc) >> filt(X.endswith('.tex')) >> one()
                            continue
                        
                        # more than one top-level tex file, keep anything with \title
                        for titledoc in ls(doc) >> filt(X.endswith('.tex')):
                            try:
                                if r'\title' in fread(titledoc):
                                    yield titledoc
                            except:
                                pass
                elif doc.endswith('.tex'):
                    yield doc

        texfiles = list(tex_files())
        pool.map(convert, texfiles)
        sh(f'mv {dump} done')
        print(f'marking {dump} as done')
    except:
        sh(f'mv {dump} errored')

pool.close()
pool.join()