def segment_text(text: str):
    import torch
    from wtpsplit import SaT
    import time
    overall_start = time.time()

   
    # Print environment information
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

    # Initialize model with GPU and half precision
    model = SaT("sat-3l-sm")
    model.half().to("cuda")

    # Process the text
    segmented = model.split(text, do_paragraph_segmentation=True, verbose=True)
    overall_end = time.time()
    print(f"\nTotal execution time: {overall_end - overall_start:.2f} seconds")
    return {"segmented_text": segmented}


print(
    segment_text(
        "Traumas dos pais a gente vai carregandoalgumas coisas que não são nossas vai setornando mais ansioso mais ruminativosno pensamentos quanto mais ansioso agente fica mais dopaminérgico mais buscapor prazer a gente tem de a ter porque agente não consegue ficar confortável noaqui e agora a gente tá sempre fugindodesse momento então eu buscava napornografia a Luxúria como eu falei umaforma de você fugir por alguns segundosné disso e a pornografia como eu jáfalei o problema dela que ela é umanovidade ilimitada e ela tem fácilacesso e é um prazer muito grande néesse instinto sexual muito forte é aúnica forma que eu consegui de combatera Luxúria mesmo né além do jejum dedopamina que eu fiz por 90 dias Cortei apornografia para ver o que aconteceriacom o meucorpo foi com empatia com relação àspessoas então eu comecei a perceber manoSerá que se eu fosse o pai dessa meninaou se eu fosse o terapeuta dela eugostaria que ela saísse com um cara quetá querendo usar ela e ser usado por elaserá que eu gostaria que ela saísse comum cara que não quer entrar entregarvalor nenhum para ela ele basicamente sóquer uma busca do parética por prazer néaquilo que eu tentava reproduzir napornografia porque é um prazer muitoforte cara é muito difícil você tentarcontrolar isso você não vai conseguir éum instinto mesmo animalesco sabe Entãodurante a adolescência consumia muitapornografia ruminava em pensamentosquanto mais você busca esse prazerpornografia drogas todo tipo que eutambém já caí em droga já usei váriostipos de droga não mergulho disso quantomais você tem essa busca dopamin éticamais o seu padrão de dopamina fica maisfásico né altos e baixos prazer prazerprazer prazer prazer prazer depressão eaí você perde controle sobre seuspensamentos tá porque o circuito dedopamina chamado mesocortical tá que édo córtex préfrontal região de autocontrole de memória de trabalho ela vaiperdendo tono de dopamina você não temcontrole de dopamina E aí você perdecontrole sobre seus pensamentos vocêfica mais ansioso por isso que porexemplo no caso do TDH que eu fuidiagnosticado TDH Qual é a comorbidademais comum que vem jun TDH né a doençamais comum que vem junto ansiedadedepois depressão por quê flutuação ehumor e para de ter controle sobre ospensamentos rumina em excesso e aí entãoeu lembro na adolescência mano eu fuibeijar pela primeira vez com 16 anosnuma festa tive que beber é isso é muitocurioso a gente precisa se intoxicarpara se desconectar para tentar seconectar com alguém né para as pessoasporque tinha uma pressão social em cimadissoe depois com 19 anos quando eu comecei asair mais cara eu lembro que eu dirigiaquilômetros mano ou sei lá dava um jeitode encontrar alguém só para ter algumprazer imediato ali e depois eu viaassim que eu tinha aquele prazer eu jávia que que eu tô fazendo aqui essabusca né Essa ânsia de ter E o tédio depossuir e aí você entender você trazeressa energia de volta para você e vocêentender que na verdade esse é umdireito inerente seu um direito denascença seu o direito de se sentir empaz a qualquer hora em qualquer lugarninguém vai te dar isso e uma forma dever isso também que é legal além daempatia né de você se colocar no lugarda pessoa e ver ela como um ser humanoentender que todos somos um só que tudovai acabar que isso é tudo ilusão é vocêimaginar depois do ato né Depois de umato de prazer seja droga seja um sexocasual que você não quer nada só querprazer seja pornografia você imaginarvocê depois como você vai sentir depoisassim que você terminar visualiza mesmosabe visualiza aquela cena eprovavelmente quando é uma buscadopaminérgica por prazer dessa formavocê vai sentir um vazio é muito comumsentir uma depressão né então quando agente deseja muito alguma coisa e vocêfinalmente consegue é comum vir seguidode uma depressão né Principalmente com éum prazer que você não teve que batalharpara conquistar ou seja né como já diriaRobert post que muito bem eu adoro essafrase dopamina não é sobre a busca doprazer é sobre o prazer da busca É sobreo processo é sobre entrar num estado deFlow isso que vai te realmente deixarfeliz e aí hoje quando eu me pego tendo"
    )
)
