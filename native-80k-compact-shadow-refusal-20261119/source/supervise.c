#define _GNU_SOURCE
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
static const char * server="/candidate/llama-server", *dso="/candidate/libggml-cuda.so";
static const uintptr_t offsets[]={0x225b230,0x225b238,0x225b240,0x225b228,0x2256c40,0x2256b4c,0x225b220};
static pid_t owned=-1;
static void fail(const char *s){perror(s);if(owned>0){kill(owned,SIGKILL);waitpid(owned,NULL,0);}exit(2);}
static bool same(struct stat a,struct stat b){return a.st_dev==b.st_dev && a.st_ino==b.st_ino;}
static bool word(int index,const char *image,const char *exe,uint64_t *out) {
    if(index<0 || index>=7 || owned<=0)return false;
    const size_t word_bytes = index == 5 ? sizeof(uint32_t) : sizeof(uint64_t);
    char p[96],line[2048];struct stat ex,loaded,elf;
    snprintf(p,sizeof p,"/proc/%d/status",owned);FILE *f=fopen(p,"r");if(!f)return false;
    int parent=-1,uids=0;unsigned u[4];
    while(fgets(line,sizeof line,f)){
        if(sscanf(line,"PPid: %d",&parent)==1)continue;
        if(sscanf(line,"Uid: %u %u %u %u",u,u+1,u+2,u+3)==4)uids=(u[0]==getuid() && u[1]==geteuid() && u[2]==geteuid() && u[3]==geteuid());
    }
    fclose(f);if(parent!=getpid() || !uids)return false;
    snprintf(p,sizeof p,"/proc/%d/exe",owned);
    if(stat(exe,&ex)||stat(p,&loaded)||!same(ex,loaded)||stat(image,&elf))return false;
    snprintf(p,sizeof p,"/proc/%d/maps",owned);f=fopen(p,"r");if(!f)return false;
    uintptr_t base=0;int bases=0;bool writable_image_end=false, private_bss=false;
    while(fgets(line,sizeof line,f)){
        unsigned long a,b,off,inode;unsigned maj,min;char perm[5];int n=0;
        if(sscanf(line,"%lx-%lx %4s %lx %x:%x %lu%n",&a,&b,perm,&off,&maj,&min,&inode,&n)!=7)continue;
        char *name=line+n;name+=strspn(name," \t");name[strcspn(name,"\n")]=0;
        bool mine=inode==elf.st_ino && makedev(maj,min)==elf.st_dev && !strcmp(name,image);
        if(mine && off==0 && perm[0]=='r'){base=a;++bases;}
        if(!base)continue;
        // ELF PT_LOAD offset 0x224deb0, vaddr 0x224eeb0, filesz 0x6900,
        // memsz 0x18b58. These pinned words live in its private zero-filled tail.
        if(mine && !strcmp(perm,"rw-p") && b==base+0x2256000 && a>=base+0x224e000 && a< b)
            writable_image_end=true;
        uintptr_t ptr=base+offsets[index];
        if(!inode && (!*name || !strcmp(name,"[anon:.bss]")) && !strcmp(perm,"rw-p") && a==base+0x2256000 && ptr>=a && ptr+word_bytes<=b &&
           ptr+word_bytes<=base+0x224eeb0+0x18b58)private_bss=true;
    }
    fclose(f);if(bases!=1 || !writable_image_end || !private_bss)return false;
    snprintf(p,sizeof p,"/proc/%d/mem",owned);
    int fd=open(p,O_RDONLY|O_CLOEXEC);if(fd<0)return false;
    *out=0;
    ssize_t bytes=pread(fd,out,word_bytes,(off_t)(base+offsets[index]));
    close(fd);return bytes==(ssize_t)word_bytes;
}
static int cpu_check(const char *image) {
    char real[PATH_MAX];if(!realpath(image,real))return 2;
    int gate[2];if(pipe(gate))return 2;
    owned=fork();if(owned<0)return 2;
    if(owned==0){
        close(gate[0]);int fd=open(real,O_RDONLY);if(fd<0)_exit(2);
        size_t n=0x2268000;char *base=mmap(NULL,n,PROT_NONE,MAP_ANONYMOUS|MAP_PRIVATE,-1,0);if(base==MAP_FAILED)_exit(2);
        if(mmap(base,4096,PROT_READ,MAP_FIXED|MAP_PRIVATE,fd,0)==MAP_FAILED ||
           mmap(base+0x224e000,0x8000,PROT_READ|PROT_WRITE,MAP_FIXED|MAP_PRIVATE,fd,0x224d000)==MAP_FAILED ||
           mmap(base+0x2256000,0x12000,PROT_READ|PROT_WRITE,MAP_FIXED|MAP_PRIVATE|MAP_ANONYMOUS,-1,0)==MAP_FAILED)_exit(2);
        for(int i=0;i<5;i++)*(uint64_t*)(base+offsets[i])=101+i;
        const uint32_t count_and_guard[]={106,UINT32_MAX};
        memcpy(base+offsets[5],count_and_guard,sizeof count_and_guard);
        *(uint64_t*)(base+offsets[6])=107;
        if(write(gate[1],"1",1)!=1)_exit(2);
        pause();_exit(0);
    }
    close(gate[1]);char c;if(read(gate[0],&c,1)!=1)fail("CPU fixture");
    for(int i=0;i<7;i++){uint64_t v=0;if(!word(i,real,"/proc/self/exe",&v)||v!=(uint64_t)(101+i))fail("CPU fixed word");}
    uint64_t v;if(word(0,"/proc/self/exe","/proc/self/exe",&v)||word(0,real,real,&v)||word(7,real,"/proc/self/exe",&v))fail("CPU false identity");
    kill(owned,SIGKILL);waitpid(owned,NULL,0);puts("CPU_PARENT_PASS seven exact words, wrong image/executable/index refused");return 0;
}
int main(int argc,char **argv){
    if(argc==3 && !strcmp(argv[1],"--cpu-check"))return cpu_check(argv[2]);
    if(argc<2 || getpid()!=1)return 2;
    sigset_t set,old;sigemptyset(&set);
    int sigs[]={SIGCHLD,SIGTERM,SIGINT,SIGUSR1};
    for(unsigned i=0;i<4;i++)sigaddset(&set,sigs[i]);
    if(sigprocmask(SIG_BLOCK,&set,&old))return 2;
    owned=fork();if(owned<0)return 2;
    if(!owned){if(sigprocmask(SIG_SETMASK,&old,NULL))_exit(2);argv[0]=(char*)server;execv(server,argv);_exit(127);}
    FILE *f=fopen("/witness/child.pid","wx");if(!f)fail("record owned child");
    fprintf(f,"%d\n",owned);if(fclose(f))fail("flush child");
    int step=0;
    for(;;){
        int sig=sigwaitinfo(&set,NULL);if(sig<0){if(errno==EINTR)continue;fail("sigwait");}
        if(sig==SIGTERM||sig==SIGINT){kill(owned,sig);continue;}
        if(sig==SIGCHLD){int status=0;pid_t r=waitpid(owned,&status,WNOHANG);if(r==0)continue;if(r!=owned)fail("wait owned");return WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);}
        if(sig!=SIGUSR1||step>=3)fail("unexpected sample request");
        uint64_t v[7];for(int i=0;i<7;i++)if(!word(i,dso,server,&v[i]))fail("owned child counter geometry");
        char dst[64];snprintf(dst,sizeof dst,"/witness/%s.json",((const char*[]){"start","cold","warm"})[step++]);
        f=fopen("/witness/pending","wx");if(!f)fail("create sample");
        fprintf(f,"{\"child\":%d,\"row0\":%"PRIu64",\"row1\":%"PRIu64",\"combined\":%"PRIu64",\"skip\":%"PRIu64",\"q2_sparse\":%"PRIu64",\"q2_stride\":%"PRIu64",\"down4\":%"PRIu64"}\n",owned,v[0],v[1],v[2],v[3],v[4],v[5],v[6]);
        if(fclose(f)||rename("/witness/pending",dst))fail("publish sample");
    }
}
