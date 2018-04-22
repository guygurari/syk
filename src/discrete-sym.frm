Functions J,Jstar,c,cd;
Indices i,j,k,l;

local [HC] = - Jstar(i,j,k,l) * (cd(i)*cd(j)*c(k)*c(l) + c(i)*c(j)*cd(k)*cd(l)) + J(i,j,k,l) * (cd(i)*cd(j)*c(k)*c(l) + c(i)*c(j)*cd(k)*cd(l)) ;

repeat;

contract;

id Jstar(i,j,k,l) = J(k,l,i,j);
id c(i?) * cd(j?) = d_(i,j) - cd(j) * c(i);
id J(i?,j?,j?,k?) = - J(j,i,j,k);
id J(j?,i?,k?,j?) = - J(j,i,j,k);

endrepeat;

repeat;
id J(k,l,i,j)*cd(k)*cd(l)*c(i)*c(j) = J(i,j,k,l)*cd(i)*cd(j)*c(k)*c(l);
id J(k,l,i,j)*cd(i)*cd(j)*c(k)*c(l) = J(i,j,k,l)*cd(k)*cd(l)*c(i)*c(j);
*id J(i?,j?,k?,j?) = J(j,i,j,k);
id J(i,l,k,l)*cd(k)*c(i) = J(i,l,j,l)*cd(j)*c(i);
id J(k,i,k,l)*cd(l)*c(i) = J(i,l,j,l)*cd(j)*c(i);
id J(k,j,k,l)*cd(l)*c(j) = J(i,l,j,l)*cd(j)*c(i);
id J(k,l,i,l)*cd(k)*c(i) = J(i,l,j,l)*cd(i)*c(j);
id J(k,l,k,i)*cd(l)*c(i) = J(i,l,j,l)*cd(i)*c(j);

id J(k,l,k,j)*cd(l)*c(j) = J(i,l,j,l)*cd(i)*c(j);
id J(l,k,l,j)*cd(k)*c(j) = J(i,l,j,l)*cd(i)*c(j);
id J(l,j,l,k)*cd(k)*c(j) = J(i,l,j,l)*cd(j)*c(i);

endrepeat;

Print;
.end

