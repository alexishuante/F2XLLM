module params
    real(8)      pi  
    parameter  ( pi = 3.1415926535897931d0 )
    real(8)      sqrpi2
    parameter  ( sqrpi2 = ( pi**( -0.5d0 ) )*2.0d0 )
    real(8)      dtol,          rcut 
    parameter  ( dtol= 1.0d-12, rcut= 1.0d-12 ) 
    real(8)      tobohrs 
    parameter  ( tobohrs = 1.889725987722d0 )
end module params

PROGRAM main
    use       params 
    implicit  none 
    integer   natom, ngauss
    integer   i
    real(8),  allocatable ::  xpnt( : )
    real(8),  allocatable ::  coef( : )
    real(8),  allocatable ::  geom( : , : )
    real(8)   j, erep
    real(8)   :: temp_xpnt(3)
    real(8)   :: temp_coef(3) 
    real(8)   :: temp_geom(24) 
    
    ! initialize input variables
    ngauss = 3
    natom = 8
    erep = 0

    ! Input data
    data temp_xpnt / 6.3624214, 1.1589230, 0.3136498 /
    data temp_coef / 0.154328967295, 0.535328142282, 0.444634542185 /
    data temp_geom / 0.0000, 0.0000, 0.0000, &
                     0.0000, 0.0000, 1.4000, &
                     0.0000, 1.4000, 0.0000, &
                     0.0000, 1.4000, 1.4000, &
                     1.4000, 0.0000, 0.0000, &   
                     1.4000, 0.0000, 1.4000, & 
                     1.4000, 1.4000, 0.0000, &       
                     1.4000, 1.4000, 1.4000 /

    ! Initialize the gaussian exponents, contraction coefficients, cartesian coordinates of each atom
    allocate( xpnt(ngauss) )
    allocate( coef( ngauss ) )  
    allocate( geom( 3, natom ) )

    xpnt = temp_xpnt
    coef = temp_coef
    do i = 1, natom*3
    j = ceiling(real(i) / 3.0)
    geom(1, i) = temp_geom(i)
    geom(2, i) = j
    end do

    ! Compute Hartree-Fock kernel
    call basic_hf_proxy(ngauss, natom, xpnt, coef, geom, erep)

    ! Print results
    print *, '2e- energy= ', erep*0.5d0  

    ! Clean up
    deallocate( xpnt )  
    deallocate( coef )  
    deallocate( geom )  
END PROGRAM main

subroutine basic_hf_proxy(ngauss, natom, xpnt, coef, geom, erep)
    use        params 
    implicit   none 
    integer    ngauss, natom
    real(8)    xpnt(*), coef(*), geom(3,*), erep
    integer    i,j,ij, ib,jb,kb,lb, nn,kl,k,l  
    real(8)    aij,dij,xij,yij,zij, akl,dkl, aijkl,tt,f0t, eri 
    integer    nnnn,ijkl,n   

    real(8),   allocatable ::  fock( : , : )
    real(8),   allocatable ::  dens( : , : )
    real(8),   allocatable ::  schwarz( : ) 
    
    ! Build density matrix from fake density 
    allocate( dens( natom, natom ) )  
    do  i  =  1,  natom  
    do  j  =  1,  natom  
    dens( i, j )  =  0.1d0   
    end do
    dens( i, i )  =  1.0d0   
    end do

    allocate( fock( natom, natom ) )  
    ! Normalize the primitive GTO weights.
    do i = 1, ngauss 
    coef( i ) = coef( i )*( ( 2.0d0*xpnt( i ) )**0.75d0 )
    end do

    !  Scale the geometry to Bohrs for energy calculations in AU.
    do  i  =  1,  natom 
    geom( 1, i )  =  geom( 1, i )*tobohrs   
    geom( 2, i )  =  geom( 2, i )*tobohrs   
    geom( 3, i )  =  geom( 3, i )*tobohrs   
    end do

    do  i  =  1,  natom  
    do  j  =  1,  natom  
    fock( i, j )  =  0.0d0  
    end do
    end do
    

    ! compute Schwarz Inequality factors for integral screening 
    nn  =  ((natom**2)+natom)/2  
    allocate( schwarz( nn ) ) 

    ij = 0 
    do  i  =  1,  natom  
    do  j  =  1,  i  
    ij = ij + 1 
    call ssss( i, j, i, j, ngauss, xpnt, coef, geom, eri )
    schwarz( ij )  =  sqrt( abs( eri ) )  
    end do
    end do

    ! Integrals are screened to avoid small terms.
    nnnn = ((nn**2)+nn)/2  

    !$OMP PARALLEL DO PRIVATE(ib,jb,kb,lb,ijkl,ij,i,j,kl,k,l,n,aij,dij,xij,yij,zij,akl,dkl,aijkl,tt,f0t,eri)

    do  ijkl  =  1,  nnnn 
    ! decompose triangular ijkl index into ij>=kl
    ij = sqrt( dble( 2*ijkl ) )
    n = ( ij*ij + ij )/2
    do  while ( n .lt. ijkl )
    ij = ij + 1
    n = ( ij*ij + ij )/2
    end do
    kl  =  ijkl - ( ij*ij - ij )/2 
    if ( schwarz( ij )*schwarz( kl ) > dtol ) then      

    ! decompose triangular ij index into i>=j
    i = sqrt( dble( 2*ij ) )
    n = ( i*i + i )/2
    do  while ( n .lt. ij )
    i = i + 1
    n = ( i*i + i )/2
    end do
    j  =  ij - ( i*i - i )/2 

    ! decompose triangular kl index into k>=l
    k = sqrt( dble( 2*kl ) )
    n = ( k*k + k )/2
    do  while ( n .lt. kl )
    k = k + 1
    n = ( k*k + k )/2
    end do
    l  =  kl - ( k*k - k )/2 

    eri  =  0.0d0 
    do  ib  =  1,  ngauss  
    do  jb  =  1,  ngauss  
    aij = 1.0d0/( xpnt( ib ) + xpnt( jb ) ) 
    dij = coef( ib )*coef( jb )*exp( -xpnt( ib )*xpnt( jb )*aij*  &  
        ( ( geom( 1, i ) - geom( 1, j ) )**2   &
        + ( geom( 2, i ) - geom( 2, j ) )**2   &
        + ( geom( 3, i ) - geom( 3, j ) )**2  )  )*( aij**1.5d0 )  
    if ( abs( dij ) > dtol ) then      
    xij = aij*( xpnt( ib )*geom( 1, i ) + xpnt( jb )*geom( 1, j ) )  
    yij = aij*( xpnt( ib )*geom( 2, i ) + xpnt( jb )*geom( 2, j ) )  
    zij = aij*( xpnt( ib )*geom( 3, i ) + xpnt( jb )*geom( 3, j ) )  
    do  kb  =  1,  ngauss  
    do  lb  =  1,  ngauss 
    akl = 1.0d0/( xpnt( kb ) + xpnt( lb ) ) 
    dkl = dij*coef( kb )*coef( lb )*exp( -xpnt( kb )*xpnt( lb )*akl*  &  
        ( ( geom( 1, k ) - geom( 1, l ) )**2   &
        + ( geom( 2, k ) - geom( 2, l ) )**2   &
        + ( geom( 3, k ) - geom( 3, l ) )**2  )  )*( akl**1.5d0 )  
    if ( abs( dkl ) > dtol ) then      
    aijkl = ( xpnt( ib ) + xpnt( jb ) )*( xpnt( kb ) + xpnt( lb ) )  &  
        / ( xpnt( ib ) + xpnt( jb )  +  xpnt( kb ) + xpnt( lb ) )  
    tt = aijkl*( ( xij -akl*( xpnt( kb )*geom( 1, k ) + xpnt( lb )*geom( 1, l ) ) )**2  & 
                + ( yij -akl*( xpnt( kb )*geom( 2, k ) + xpnt( lb )*geom( 2, l ) ) )**2  & 
                + ( zij -akl*( xpnt( kb )*geom( 3, k ) + xpnt( lb )*geom( 3, l ) ) )**2  ) 
    f0t  =  sqrpi2 
    if ( tt > rcut )  f0t  =  ( tt**( -0.5d0 ) )*erf( sqrt(tt) ) 
    eri  =  eri  +  dkl*f0t*sqrt(aijkl)  
    end if 
    end do  ;  end do  
    end if  
    end do  ;  end do 

    if ( i == j ) eri = eri*0.5d0 
    if ( k == l ) eri = eri*0.5d0 
    if ( i == k .and. j == l ) eri = eri*0.5d0
    !$OMP ATOMIC
    fock( i, j )  =  fock( i, j )  +dens( k, l )*eri*4.0d0  
    !$OMP ATOMIC
    fock( k, l )  =  fock( k, l )  +dens( i, j )*eri*4.0d0  
    !$OMP ATOMIC
    fock( i, k )  =  fock( i, k )  -dens( j, l )*eri  
    !$OMP ATOMIC
    fock( i, l )  =  fock( i, l )  -dens( j, k )*eri  
    !$OMP ATOMIC
    fock( j, k )  =  fock( j, k )  -dens( i, l )*eri  
    !$OMP ATOMIC
    fock( j, l )  =  fock( j, l )  -dens( i, k )*eri  

    end if  
    end do  
    !$OMP END PARALLEL DO

    !  Trace Fock with the density and print the 2e- energy.
    erep  =  0.0d0
    do  i  =  1,  natom  
    do  j  =  1,  natom 
    erep  =  erep  +  fock( i, j )*dens( i, j ) 
    end do
    end do 

    deallocate( dens )  
    deallocate( fock ) 
    deallocate( schwarz ) 
end subroutine basic_hf_proxy

subroutine ssss( i, j, k, l, ngauss, xpnt, coef, geom, eri )
    use        params 
    implicit   none 
    integer    i,j,k,l, ngauss 
    real(8)    xpnt(*), coef(*), geom(3,*), eri 
    integer    ib,jb,kb,lb 
    real(8)    aij,dij,xij,yij,zij, akl,dkl, aijkl,tt,f0t  

    !  loops over contracted GTOs  

    eri  =  0.0d0 
    do  ib  =  1,  ngauss  
    do  jb  =  1,  ngauss  
    aij = 1.0d0/( xpnt( ib ) + xpnt( jb ) ) 
    dij = coef( ib )*coef( jb )*exp( -xpnt( ib )*xpnt( jb )*aij*  &  
        ( ( geom( 1, i ) - geom( 1, j ) )**2   &
        + ( geom( 2, i ) - geom( 2, j ) )**2   &
        + ( geom( 3, i ) - geom( 3, j ) )**2  )  )*( aij**1.5d0 )  
    if ( abs( dij ) > dtol ) then      
    xij = aij*( xpnt( ib )*geom( 1, i ) + xpnt( jb )*geom( 1, j ) )  
    yij = aij*( xpnt( ib )*geom( 2, i ) + xpnt( jb )*geom( 2, j ) )  
    zij = aij*( xpnt( ib )*geom( 3, i ) + xpnt( jb )*geom( 3, j ) )  
    do  kb  =  1,  ngauss  
    do  lb  =  1,  ngauss 
    akl = 1.0d0/( xpnt( kb ) + xpnt( lb ) ) 
    dkl = dij*coef( kb )*coef( lb )*exp( -xpnt( kb )*xpnt( lb )*akl*  &  
        ( ( geom( 1, k ) - geom( 1, l ) )**2   &
        + ( geom( 2, k ) - geom( 2, l ) )**2   &
        + ( geom( 3, k ) - geom( 3, l ) )**2  )  )*( akl**1.5d0 )  
    if ( abs( dkl ) > dtol ) then      
    aijkl = ( xpnt( ib ) + xpnt( jb ) )*( xpnt( kb ) + xpnt( lb ) )  &  
        / ( xpnt( ib ) + xpnt( jb )  +  xpnt( kb ) + xpnt( lb ) )  
    tt = aijkl*( ( xij -akl*( xpnt( kb )*geom( 1, k ) + xpnt( lb )*geom( 1, l ) ) )**2  & 
                + ( yij -akl*( xpnt( kb )*geom( 2, k ) + xpnt( lb )*geom( 2, l ) ) )**2  & 
                + ( zij -akl*( xpnt( kb )*geom( 3, k ) + xpnt( lb )*geom( 3, l ) ) )**2  ) 
    f0t  =  sqrpi2 
    if ( tt > rcut )  f0t  =  ( tt**( -0.5d0 ) )*erf( sqrt(tt) ) 
    eri  =  eri  +  dkl*f0t*sqrt(aijkl)  
    end if 
    end do  ;  end do  
    end if  
    end do  ;  end do 
 end subroutine ssss
